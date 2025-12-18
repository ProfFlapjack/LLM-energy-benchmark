import base64
import csv
import io
import json
import os
import shutil
import time
import gc
from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg", force=True) 
import matplotlib.pyplot as plt
import pandas as pd

from codecarbon import EmissionsTracker
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from llama_cpp import Llama

app = FastAPI()

#connects the frontend to this backend
origins = [
    "http://localhost:5173"
]

#This will protect our server
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    name: str
    prompt: str
    answer: str

# this will not persist when the app shuts down
promptsList_db = {"prompts": []}

#setting the csv file for the prompts
@app.post('/uploadpromptfile/')
async def upload_prompt_file(file_upload: UploadFile = File(...)):
    data = await file_upload.read()
    text = data.decode("utf-8")

    # Clear old prompts
    promptsList_db["prompts"] = []

    errors = []

    # Parse CSV in memory
    reader = csv.DictReader(io.StringIO(text))
    for i, row in enumerate(reader, start=2):  # start=2 because row 1 is header
        try:
            prompt_obj = Prompt(**row)
            promptsList_db["prompts"].append(prompt_obj)
        except ValidationError as e:
            errors.append({"line": i, "error": e.errors(), "row": row})

    return {
        "message": "Prompts loaded",
        "count": len(promptsList_db["prompts"]),
        "errors": errors[:25],  # don’t spam huge responses
        "error_count": len(errors),
    }

# Models upload/list/delete
UPLOAD_DIR = Path("uploaded_models")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def safe_model_path(filename: str) -> Path:
    return UPLOAD_DIR / Path(filename).name

# allows for the front end to insert models
@app.post("/upload-models")
async def upload_models(
    models: List[UploadFile] = File(...),
    metadata: str | None = None,
):
    parsed_metadata = json.loads(metadata) if metadata else None
    saved_files = []

    for model_file in models:
        file_location = safe_model_path(model_file.filename)
        with file_location.open("wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
        saved_files.append(str(file_location))

    return {
            "count": len(models), 
            "files": saved_files, 
            "metadata": parsed_metadata
        }


# Allows for the front end to see what models are in the backend
@app.get("/models")
def list_models():
    models = []
    for p in sorted(UPLOAD_DIR.iterdir()):
        if p.is_file() and p.name.lower().endswith(".gguf"):
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            models.append({"filename": p.name, "path": str(p), "size_mb": size_mb})
    return {
            "models": models, 
            "count": len(models)
        }

# allows for the front end to delete models
@app.delete("/models/{filename}")
def delete_model(filename: str):
    path = safe_model_path(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Model not found")
    path.unlink()
    return {"deleted": path.name}



#graph creation
#Since the website cant take in figures, have to convert the figures into images
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # free memory
    return img_b64

# THis is the main structure to make the bar plots look nice
def make_bar_plot(labels, values, title, xlabel):
    # sort by value (ascending)
    pairs = sorted(zip(values, labels))
    values, labels = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values)

    ax.set_title(title)
    ax.set_xlabel(xlabel)

    ax.bar_label(bars, fmt="%.3e", label_type="center", fontsize=9)

    fig.tight_layout()
    return fig_to_base64(fig)

def make_bar_plot_for_prompts(labels, values, title, xlabel):
    # sort by value (ascending)
    pairs = sorted(zip(values, labels))
    values, labels = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values)

    ax.set_title(title)
    ax.set_xlabel(xlabel)

    fig.tight_layout()
    return fig_to_base64(fig)


####################
# Globals
####################
llm: Optional[Llama] = None
active_model: Optional[str] = None
active_model_path: Optional[Path] = None

#keeps track of how far the progress gets
# Progress state 
benchmark_progress = {
    "current": 0,
    "total": 0,
    "status": "idle",  # idle | running | done | error
}


####################
# Utilities
####################
def normalize(s: str) -> str:
    return (s or "").strip().lower()

def list_model_files(upload_dir: Path) -> list[Path]:
    return sorted([
        p for p in upload_dir.iterdir()
        if p.is_file() and p.name.lower().endswith(".gguf")
    ])

def load_model_from_path(model_path: Path, n_ctx: int = 2048, n_threads: int = 8):
    """Loads a GGUF model and sets globals."""
    global llm, active_model, active_model_path

    llm = None
    gc.collect()

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
    )
    active_model = model_path.name
    active_model_path = model_path


    return llm

def ensure_fresh_state(n_ctx: int, n_threads: int):
    """
    Ensure clean KV-cache for each prompt.
    Prefer llm.reset() if available; otherwise reload model (slower but correct).
    """
    global llm, active_model_path
    # best case
    if llm is not None and hasattr(llm, "reset") and callable(llm.reset):
        # print("ensure_fresh_state: used llm.reset()")
        llm.reset()
        return
    # print("ensure_fresh_state: reloading model (no reset)")
    # fallback: reload model (correct but slower)
    if active_model_path is None:
        raise RuntimeError("No active model path set")
    load_model_from_path(active_model_path, n_ctx=n_ctx, n_threads=n_threads)



####################
# Core measurement
####################
def run_with_token_level_lists(
    #variables to store the prompt and expected answer
    prompt: str,
    expected: str,
    max_tokens: int = 64,
):
    """
    Returns:
      - prefill_times: per prompt token time (true token-by-token eval)
      - decode_times: per generated token time (time between token yields; proxy)
      - energy_prefill_kwh, energy_decode_kwh, total
    """
    if llm is None:
        raise RuntimeError("No model loaded")

    # Tokenize prompt
    try:
        prompt_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    except TypeError:
        prompt_tokens = llm.tokenize(prompt.encode("utf-8"))

    n_prompt = len(prompt_tokens)

    ############## PREFILL (token-by-token eval) ###########
    prefill_times = []
    tracker_prefill = EmissionsTracker(save_to_file=False, log_level="error")
    tracker_prefill.start()

    for tok in prompt_tokens:
        t0 = time.perf_counter()
        llm.eval([tok])
        t1 = time.perf_counter()
        prefill_times.append(t1 - t0)

    tracker_prefill.stop()
    energy_prefill_kwh = tracker_prefill.final_emissions_data.energy_consumed

    ############### DECODE (per-token yield latency proxy) ############
    decode_times = []
    generated_token_ids = []

    tracker_decode = EmissionsTracker(save_to_file=False, log_level="error")
    tracker_decode.start()

    last_t = time.perf_counter()

    # Try to continue from prefill state (KV cache) if reset= is supported
    try:
        gen_iter = llm.generate(
            tokens=[],          # continue from current state (after your llm.eval prompt tokens)
            temp=0.0,
            top_k=1,
            top_p=1.0,
            reset=False,        # some versions support this
        )
    except TypeError:
        # fallback if reset= isn't supported in your generate()
        gen_iter = llm.generate(
            tokens=[],
            temp=0.0,
            top_k=1,
            top_p=1.0,
        )

    # It iterates over generated tokens, tracks how long each token takes to generate, stores their IDs, and stops after generating a set number of tokens
    for tok in gen_iter:
        now = time.perf_counter()

        decode_times.append(now - last_t)
        # starts new timer
        last_t = now

        generated_token_ids.append(tok)

        if len(generated_token_ids) >= max_tokens:
            break

    tracker_decode.stop()
    energy_decode_kwh = tracker_decode.final_emissions_data.energy_consumed



    # Decode output
    output_text = llm.detokenize(generated_token_ids).decode("utf-8", errors="ignore")

    ok = normalize(expected) in normalize(output_text)

    prefill_time = sum(prefill_times)
    decode_time = sum(decode_times)
    total_time = prefill_time + decode_time

    n_gen = len(generated_token_ids)

    # Energy per token
    e_prefill_per_tok = energy_prefill_kwh / n_prompt if n_prompt else 0.0
    e_decode_per_tok  = energy_decode_kwh / n_gen if n_gen else 0.0

    

    # Cumulative curves (for charts)
    prefill_cum = []
    running = 0.0
    for dt in prefill_times:
        running += dt
        prefill_cum.append(running)

    decode_cum = []
    running = 0.0
    for dt in decode_times:
        running += dt
        decode_cum.append(running)

    # returns all the statistics collected
    return {
        "prompt_tokens": n_prompt,
        "generated_tokens": n_gen,

        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,

        "energy_prefill_kwh": energy_prefill_kwh,
        "energy_decode_kwh": energy_decode_kwh,
        "energy_total_kwh": energy_prefill_kwh + energy_decode_kwh,

        "energy_per_prompt_token_kwh": e_prefill_per_tok,
        "energy_per_generated_token_kwh": e_decode_per_tok,

        "prefill_cumulative": prefill_cum,
        "decode_cumulative": decode_cum,

        "answer_match": ok,
    }


####################
# Benchmark runners
####################
def run_benchmark_for_loaded_model(
    max_tokens: int = 64,
    n_ctx: int = 2048,
    n_threads: int = 8,
):
    # checks if everthing is loaded
    if llm is None:
        raise RuntimeError("No model loaded")
    if not promptsList_db["prompts"]:
        raise RuntimeError("No prompts uploaded")

    results = []

    #Goes through each prompt and records stats as in time and energy cusumption
    for row in promptsList_db["prompts"]:
        # critical to get clean KV-cache per prompt
        #This refreshes the model to make sure it is not useing previous models for context
        ensure_fresh_state(n_ctx=n_ctx, n_threads=n_threads)

        #This runs the model directly with the promp and answer
        stats = run_with_token_level_lists(
            prompt=row.prompt,
            expected=row.answer,
            max_tokens=max_tokens,
        )

        benchmark_progress["current"] += 1

        results.append({
            "name": row.name,
            "expected": row.answer,
            **stats,
        })

    total = len(results)
    correct = sum(1 for r in results if r["answer_match"])

    # Aggregate model-level totals
    total_energy = sum(r["energy_total_kwh"] for r in results)
    total_time = sum(r["total_time"] for r in results)

    #returns stats
    return {
        "model": active_model,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "total_energy_kwh": total_energy,
        "total_time_s": total_time,
        "results": results,
    }

#Method to run all of these models and keep track
# This method fetches the prompts and model files, loads them and then sends it to a method to be measured
#This is where most of the important code is
def run_all_models(
    upload_dir: Path,
    max_tokens: int = 64,
    n_ctx: int = 2048,
    n_threads: int = 8,
):
    # makes sure everthing is loaded
    if not promptsList_db["prompts"]:
        raise RuntimeError("No prompts uploaded")

    model_files = list_model_files(upload_dir)
    if not model_files:
        raise RuntimeError("No .gguf models found in uploaded_models/")

    all_model_results = []
    # this is for the progress bar
    benchmark_progress["current"] = 0
    benchmark_progress["total"] = len(model_files) * len(promptsList_db["prompts"])
    benchmark_progress["status"] = "running"

    # loads each model and warms it up with a hello
    for model_path in model_files:
        load_model_from_path(model_path, n_ctx=n_ctx, n_threads=n_threads)

        _ = llm("Hello", max_tokens=1, temperature=0.0)  # warmup
        # after warm up sends it to be measured on other prompts
        model_result = run_benchmark_for_loaded_model(
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            n_threads=n_threads,
        )
        all_model_results.append(model_result)

    #### BUILD SUMMARY SERIES (one value per model)
    labels = [r["model"] for r in all_model_results]

    # examples (adjust keys to your actual returned dicts)
    accuracy_vals = [r["accuracy"] for r in all_model_results]

    # total energy/time per model (sum over prompts)
    total_time_vals = [sum(x["total_time"] for x in r["results"]) for r in all_model_results]
    total_energy_vals = [sum(x.get("energy_total_kwh", 0.0) for x in r["results"]) for r in all_model_results]

    # energy per token (avg over prompts)
    time_per_prompt_token_vals = []
    time_per_generated_token_vals = []



    prefill_ept = []
    decode_ept = []
    #gets the sum over everything per model
    for r in all_model_results:
        prefill_energy = sum(x.get("energy_prefill_kwh", 0.0) for x in r["results"])
        decode_energy  = sum(x.get("energy_decode_kwh", 0.0) for x in r["results"])

        prefill_time   = sum(x.get("prefill_time", 0.0) for x in r["results"])
        decode_time    = sum(x.get("decode_time", 0.0) for x in r["results"])

        prompt_toks    = sum(x.get("prompt_tokens", 0) for x in r["results"])
        gen_toks       = sum(x.get("generated_tokens", 0) for x in r["results"])
        
        prefill_ept.append(prefill_energy / prompt_toks if prompt_toks else 0.0)
        decode_ept.append(decode_energy / gen_toks if gen_toks else 0.0)

        time_per_prompt_token_vals.append(prefill_time / prompt_toks if prompt_toks else 0.0)
        time_per_generated_token_vals.append(decode_time / gen_toks if gen_toks else 0.0)


        

    #### CREATE PLOTS (base64)
    plots = {
        "accuracy_bar": make_bar_plot(labels, accuracy_vals, "Accuracy by model", "Accuracy"),
        "energy_per_prompt_token_bar": make_bar_plot(labels, prefill_ept, "Energy per prompt token", "kWh / token"),
        "energy_per_generated_token_bar": make_bar_plot(labels, decode_ept, "Energy per generated token", "kWh / token"),
        "time_per_prompt_token_bar": make_bar_plot(labels, time_per_prompt_token_vals, "Time per prompt token", "seconds / token"),
        "time_per_generated_token_bar": make_bar_plot(labels, time_per_generated_token_vals, "Time per generated token", "seconds / token"),

        "total_energy_bar": make_bar_plot(labels, total_energy_vals, "Total energy", "kWh"),
        "total_time_bar": make_bar_plot(labels, total_time_vals, "Total time", "seconds"),
        

    }

    #######################################
    # Per-prompt plots (across models)
    #######################################

    # Get the prompt order from the uploaded CSV (stable ordering)
    prompt_names = [p.name for p in promptsList_db["prompts"]]

    per_model_prompt_correctness_plots = {}
    per_model_prompt_time_plots = {}

    for model_run in all_model_results:
        model_name = model_run["model"]

        # ---- Correctness (0/1) per prompt for THIS model
        match_map = {
            r["name"]: (1.0 if r.get("answer_match") else 0.0)
            for r in model_run["results"]
        }
        correctness_vals = [match_map.get(name, 0.0) for name in prompt_names]

        per_model_prompt_correctness_plots[model_name] = make_bar_plot_for_prompts(
            prompt_names,
            correctness_vals,
            title=f"Correctness by Prompt (0/1) - {model_name}",
            xlabel="Correct (1) / Incorrect (0)"
        )

        # ---- Total time per prompt for THIS model
        time_map = {
            r["name"]: float(r.get("total_time", 0.0))
            for r in model_run["results"]
        }
        time_vals = [time_map.get(name, 0.0) for name in prompt_names]

        per_model_prompt_time_plots[model_name] = make_bar_plot_for_prompts(
            prompt_names,
            time_vals,
            title=f"Total Time by Prompt (s) - {model_name}",
            xlabel="Seconds"
        )


    benchmark_progress["status"] = "done"

    return {
        "models_tested": len(all_model_results),
        "results": all_model_results,
        "plots": plots,
        "per_model_plots": {
            "correctness_by_prompt": per_model_prompt_correctness_plots,
            "time_by_prompt": per_model_prompt_time_plots,
        },
        "summary": {
            "labels": labels,
            "accuracy": accuracy_vals,
            "total_time": total_time_vals,
            "total_energy_kwh": total_energy_vals,
            "energy_per_prompt_token_kwh": prefill_ept,
            "energy_per_generated_token_kwh": decode_ept,
        },
    }


@app.get("/benchmark_progress")
def get_benchmark_progress():
    return benchmark_progress

@app.get("/prompt_status")
def prompt_status():
    return {
        "count": len(promptsList_db["prompts"])
    }


#This is connected to the run button and executes tests on all the models 
@app.post("/run_all_tests")
def run_all_tests(max_tokens: int = 64, n_ctx: int = 2048, n_threads: int = 8):
    # starts a progress bar to see how far the models have progressed
    benchmark_progress["current"] = 0
    benchmark_progress["total"] = 0
    benchmark_progress["status"] = "running"

    # uses run all models method to run the models
    # It will return plots and stats for the models, then these will be sent to the frontend
    try:
        return run_all_models(
            upload_dir=UPLOAD_DIR,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            n_threads=n_threads,
        )
    except Exception as e:
        benchmark_progress["status"] = "error"
        import traceback
        traceback.print_exc()  
        raise HTTPException(status_code=400, detail=str(e))

#  LLM Quantization Benchmarking Framework

This project provides an interactive, fully offline benchmarking framework to evaluate **energy consumption**, **inference time**, and **accuracy** across quantized Large Language Models (LLMs) from the LLaMA 3.1 family in GGUF format.

---

##  Features

- Upload and benchmark multiple quantized GGUF models locally
- Evaluate models on custom prompt datasets
- Track estimated energy usage using [CodeCarbon](https://mlco2.github.io/codecarbon/)
- Display visual benchmarks (latency, energy, accuracy)
- Fully offline: CPU-only inference with reproducible setup

---

##  System Overview

- **Frontend**: React-based web interface for configuration and visualization
- **Backend**: FastAPI server that manages benchmarking and metrics
- **Model Execution**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (wrapped GGUF model inference)

---

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/profflapjack/your-repo-name.git
cd your-repo-name
```


### 2. Backend Setup (Python 3.10+ Recommended)
In command prompt:
```
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
You will have to run the activate script and requirments if you go out of the terminal
### 3. Frontend Setup
Open another terminal and in that terminal type:
```
cd frontend
npm install
npm run build
```
### 4. Running the app
For front end:
``` 
uvicorn main:app --reload
```
For back end:
```
npm run dev 
```

Then open the url for the front end

### 5. Upload models and prompt csv
Models have to be in the gguf format

The csv file needs to follow a similar pattern using name, prompt, and answer
Example:
```
name,prompt,answer
math_small_001,"What is 17 × 24?","408"
math_medium_001,"Calculate the result of multiplying 17 by 24.","408"
math_large_001,"Please solve the following arithmetic problem carefully. What is the product of 17 multiplied by 24?","408"

logic_small_002,"True or False: If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?","True"
logic_medium_002,"True or False: Consider the following statements: All bloops are razzies. All razzies are lazzies. Based on this information, are all bloops lazzies?","True"
logic_large_002,"True or False: You are given a logical reasoning problem. All bloops belong to the category of razzies, and all razzies belong to the category of lazzies. Using deductive reasoning, determine whether it necessarily follows that all bloops are lazzies.","True"

fact_small_003,"What is the capital of France?","Paris"
fact_medium_003,"Identify the capital city of the country France.","Paris"
fact_large_003,"France is a country located in Western Europe. What is the name of its capital city, which also serves as a major cultural, political, and economic center?","Paris"
```


## Notes!
Uploading multiple models at once may take some time! It is not frozen but taking in each model.

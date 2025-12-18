import axios from 'axios';

//create and instance of axios with the base URL
const api = axios.create({
    baseURL: "http://localhost:8000"
});

//export the axios instance
export default api;
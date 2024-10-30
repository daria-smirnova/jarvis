import json
import requests
import uuid
import base64
import random
import threading
import time
import traceback
from io import BytesIO
from PIL import Image, ImageDraw
from pydub import AudioSegment
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# Global configurations (modify as needed)
Model_Server = "http://localhost:8000"  # URL for local model server
HUGGINGFACE_HEADERS = {"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"}
PROXY = {}
API_KEY = ""
API_TYPE = "huggingface"
API_ENDPOINT = "your_api_endpoint"
inference_mode = "local"  # or "huggingface"
config = {
    "local_deployment": "full",
    "num_candidate_models": 5,
    "max_description_length": 100
}
MODELS_MAP = {
    "summarization": [{"id": "summarization_model_1"}],
    "translation": [{"id": "translation_model_1"}],
    "conversational": [{"id": "conversational_model_1"}],
    "text-generation": [{"id": "text_generation_model_1"}],
    "text2text-generation": [{"id": "text2text_generation_model_1"}],
    # Add other task types and models here
}


def inference(inputs, raw_response=False):
    """Placeholder function for model inference."""
    # Here you should call the model inference API or local model
    # For now, return a dummy response
    return requests.Response()


def local_model_inference(model_id, data, task):
    task_url = f"{Model_Server}/models/{model_id}"

    if model_id.startswith("lllyasviel/sd-controlnet-"):
        img_url = data["image"]
        text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results

    if model_id.endswith("-control"):
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results

    if task == "text-to-video":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated video"] = results.pop("path")
        return results

    if task in ["question-answering", "sentence-similarity"]:
        response = requests.post(task_url, json=data)
        return response.json()

    if task in ["text-classification", "token-classification", "text2text-generation", "summarization", "translation",
                "conversational", "text-generation"]:
        response = requests.post(task_url, json=data)
        return response.json()

    if task == "depth-estimation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results

    if task == "image-segmentation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        results["generated image"] = results.pop("path")
        return results

    if task == "image-to-image":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results

    if task == "text-to-image":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results

    if task == "object-detection":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        predicted = response.json()
        if "error" in predicted:
            return predicted
        image = Image.open(requests.get(img_url, stream=True).raw)
        draw = ImageDraw.Draw(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]],
                           width=2)
            draw.text((box["xmin"] + 5, box["ymin"] - 15), label["label"], fill=color_map[label["label"]])
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        results = {}
        results["generated image"] = f"/images/{name}.jpg"
        results["predicted"] = predicted
        return results

    if task in ["image-classification", "image-to-text", "document-question-answering", "visual-question-answering"]:
        img_url = data["image"]
        text = None
        if "text" in data:
            text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()
        return results

    if task == "text-to-speech":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated audio"] = results.pop("path")
        return results

    if task in ["automatic-speech-recognition", "audio-to-audio", "audio-classification"]:
        audio_url = data["audio"]
        response = requests.post(task_url, json={"audio_url": audio_url})
        return response.json()


def model_inference(model_id, data, hosted_on, task):
    if hosted_on == "unknown":
        localStatusUrl = f"{Model_Server}/status/{model_id}"
        r = requests.get(localStatusUrl)
        if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
            hosted_on = "local"
        else:
            huggingfaceStatusUrl = f"https://api-inference.huggingface.co/status/{model_id}"
            r = requests.get(huggingfaceStatusUrl, headers=HUGGINGFACE_HEADERS, proxies=PROXY)
            if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
                hosted_on = "huggingface"
    try:
        if hosted_on == "local":
            inference_result = local_model_inference(model_id, data, task)
        elif hosted_on == "huggingface":
            inference_result = huggingface_model_inference(model_id, data, task)
    except Exception as e:
        print(e)
        traceback.print_exc()
        inference_result = {"error": {"message": str(e)}}
    return inference_result


def get_model_status(model_id, url, headers, queue=None):
    endpoint_type = "huggingface" if "huggingface" in url else "local"
    if "huggingface" in url:
        r = requests.get(url, headers=headers, proxies=PROXY)
    else:
        r = requests.get(url)
    if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
        if queue:
            queue.put((model_id, True, endpoint_type))
        return True
    else:
        if queue:
            queue.put((model_id, False, None))
        return False


def get_avaliable_models(candidates, topk=5):
    all_available_models = {"local": [], "huggingface": []}
    threads = []
    result_queue = Queue()

    for candidate in candidates:
        model_id = candidate["id"]

        if inference_mode != "local":
            huggingfaceStatusUrl = f"https://api-inference.huggingface.co/status/{model_id}"
            thread = threading.Thread(target=get_model_status,
                                      args=(model_id, huggingfaceStatusUrl, HUGGINGFACE_HEADERS, result_queue))
            threads.append(thread)
            thread.start()

        if inference_mode != "huggingface" and config["local_deployment"] != "minimal":
            localStatusUrl = f"{Model_Server}/status/{model_id}"
            thread = threading.Thread(target=get_model_status, args=(model_id, localStatusUrl, {}, result_queue))
            threads.append(thread)
            thread.start()

    result_count = len(threads)
    while result_count:
        model_id, status, endpoint_type = result_queue.get()
        if status:
            all_available_models[endpoint_type].append(model_id)
        result_count -= 1

    for thread in threads:
        thread.join()

    selected_models = {endpoint: random.sample(models, min(topk, len(models))) for endpoint, models in
                       all_available_models.items()}

    return selected_models


def chat_vicuna(messages, api_key, api_type, api_endpoint, return_planning=False, return_results=False):
    start = time.time()
    context = messages[:-1]
    input = messages[-1]["content"]

    task_str = parse_task(context, input, api_key, api_type, api_endpoint)

    if "error" in task_str:
        return {"message": task_str["error"]["message"]}

    task_str = task_str.strip()
    try:
        tasks = json.loads(task_str)
    except Exception as e:
        response = chitchat(messages, api_key, api_type, api_endpoint)
        return {"message": response}

    if task_str == "[]":
        response = chitchat(messages, api_key, api_type, api_endpoint)
        return {"message": response}

    if len(tasks) == 1 and tasks[0]["task"] in ["summarization", "translation", "conversational", "text-generation",
                                                "text2text-generation"]:
        response = chitchat(messages, api_key, api_type, api_endpoint)
        return {"message": response}

    tasks = unfold(tasks)
    tasks = fix_dep(tasks)

    if return_planning:
        return tasks

    results = {}
    threads = []
    tasks = tasks[:]
    d = dict()
    retry = 0
    while True:
        num_thread = len(threads)
        for task in tasks:
            for dep_id in task["dep"]:
                if dep_id >= task["id"]:
                    task["dep"] = [-1]
                    break
            dep = task["dep"]
            if dep[0] == -1 or len(list(set(dep).intersection(d.keys()))) == len(dep):
                tasks.remove(task)
                thread = threading.Thread(target=run_task, args=(input, task, d, api_key, api_type, api_endpoint))
                thread.start()
                threads.append(thread)
        if num_thread == len(threads):
            time.sleep(0.5)
            retry += 1
        if retry > 160:
            break
        if len(tasks) == 0:
            break
    for thread in threads:
        thread.join()

    results = d.copy()

    if return_results:
        return results

    response = response_results(input, results, api_key, api_type, api_endpoint).strip()

    end = time.time()
    during = end - start

    return {"message": response}


def parse_task(context, input, api_key, api_type, api_endpoint):
    # Implement this function based on your requirements
    return "[]"  # Placeholder return


def run_task(input, task, result_dict, api_key, api_type, api_endpoint):
    # Implement this function to run the task
    pass


def response_results(input, results, api_key, api_type, api_endpoint):
    # Implement this function to format the results
    return "Response results"


def chitchat(messages, api_key, api_type, api_endpoint):
    # Implement this function based on your chatbot logic
    return "Chitchat response"


def server():
    app = Flask(__name__, static_folder="public", static_url_path="/")
    app.config['DEBUG'] = False
    CORS(app)

    @cross_origin()
    @app.route('/tasks', methods=['POST'])
    def tasks():
        data = request.get_json()
        messages = data["messages"]
        api_key = data.get("api_key", API_KEY)
        api_endpoint = data.get("api_endpoint", API_ENDPOINT)
        api_type = data.get("api_type", API_TYPE)
        if api_key is None or api_type is None or api_endpoint is None:
            return jsonify({"error": "Please provide api_key, api_type and api_endpoint"})
        response = chat_vicuna(messages, api_key, api_type, api_endpoint, return_planning=True)
        return jsonify(response)

    @cross_origin()
    @app.route('/results', methods=['POST'])
    def results():
        data = request.get_json()
        messages = data["messages"]
        api_key = data.get("api_key", API_KEY)
        api_endpoint = data.get("api_endpoint", API_ENDPOINT)
        api_type = data.get("api_type", API_TYPE)
        if api_key is None or api_type is None or api_endpoint is None:
            return jsonify({"error": "Please provide api_key, api_type and api_endpoint"})
        response = chat_vicuna(messages, api_key, api_type, api_endpoint, return_results=True)
        return jsonify(response)

    app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    server()

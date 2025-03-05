import json

import numpy as np
from sentence_transformers import SentenceTransformer

class JSONLoader:
    _instance = None  # Class-level variable to store the single instance

    def __new__(cls, file_path):
        """Ensure only one instance of JSONLoader exists."""
        if cls._instance is None:
            cls._instance = super(JSONLoader, cls).__new__(cls)
            cls._instance._initialized = False  # Prevent multiple initializations
        return cls._instance

    def __init__(self, file_path):
        """Initialize only once even if called multiple times."""
        if not self._initialized:
            self.file_path = file_path
            self.data = self.load_json()
            self._initialized = True  # Mark as initialized

    def load_json(self):
        """Load the JSON file and return its contents."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: File '{self.file_path}' contains invalid JSON.")
            return None

    def get_data(self):
        """Return the loaded JSON data."""
        return self.data


def get_most_suitable_embedding(prompt, embedding_file='app/resources/postman_embeddings.json', model_name='all-MiniLM-L6-v2'):
    loaded_data =  JSONLoader(embedding_file).data
    # Determine the structure and extract embeddings and texts
    if isinstance(loaded_data, list) and len(loaded_data) > 0:
        # Check if the first element is a dictionary with an "embedding" key.
        if isinstance(loaded_data[0], dict) and 'text_embeddings' in loaded_data[0]:
            # Extract embeddings and corresponding texts (using 'text' or 'name')
            data_texts = [entry.get('text', entry.get('name', '')) for entry in loaded_data]
            text_embeddings = [entry['text_embeddings'] for entry in loaded_data]
            name_embeddings = [entry['name_embeddings'] for entry in loaded_data]
            curl_embeddings = [entry['curl_embeddings'] for entry in loaded_data]
        else:
            raise ValueError("JSON file does not contain the expected dictionary structure with an 'embedding' key.")
    else:
        raise ValueError("The loaded JSON is empty or not a list.")

    # Convert embeddings to a NumPy array for computation
    return get_top_5_similarities(text_embeddings, name_embeddings, curl_embeddings, data_texts, prompt, model_name)

def get_top_5_similarities(text_embeddings, name_embeddings, curl_embeddings, data_texts, prompt, model_name):
    # Convert embeddings to a NumPy array for computation
    text_embeddings = np.array(text_embeddings)
    name_embeddings = np.array(name_embeddings)
    curl_embeddings = np.array(curl_embeddings)

    # Initialize the SentenceTransformer model and compute the prompt's embedding
    model = SentenceTransformer(model_name)
    prompt_embedding = model.encode(prompt)

    # Compute cosine similarities between the prompt and each stored embedding.
    prompt_norm = np.linalg.norm(prompt_embedding)
    text_embeddings_norm = np.linalg.norm(text_embeddings, axis=1)
    text_cos_similarities = np.dot(text_embeddings, prompt_embedding) / (text_embeddings_norm * prompt_norm)

    name_embeddings_norm = np.linalg.norm(name_embeddings, axis=1)
    name_cos_similarities = np.dot(name_embeddings, prompt_embedding) / (name_embeddings_norm * prompt_norm)

    curl_embeddings_norm = np.linalg.norm(curl_embeddings, axis=1)
    curl_cos_similarities = np.dot(curl_embeddings, prompt_embedding) / (curl_embeddings_norm * prompt_norm)

    stacked = np.stack([text_cos_similarities, name_cos_similarities, curl_cos_similarities], axis=1)
    flat_similarities = stacked.flatten()
    top_5_indices = np.argsort(flat_similarities)[-5:][::-1]  # Get indices of top 5 values

    top_5_results = []
    for index in top_5_indices:
        arr = np.unravel_index(index, stacked.shape)
        best_text = data_texts[arr[0]]
        best_similarity = float(flat_similarities[index])
        top_5_results.append({'text': best_text, 'similarity': best_similarity})

    return top_5_results

def create_postman_embeddings(input_file='app/resources/greytHR-API-V2.postman_collection.json',
                              output_file='app/resources/postman_embeddings.json',
                              model_name='all-MiniLM-L6-v2'):
    # Load the Postman collection.
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract endpoints; Postman collections have a top-level 'item' key.
    items = data.get('item', [])
    endpoints = []
    extract_endpoints(items, endpoints)

    # Extract texts from endpoints for embedding.
    texts = [endpoint['text'] for endpoint in endpoints]
    names = [endpoint['name'] for endpoint in endpoints]
    curls = [endpoint['curl'] for endpoint in endpoints]
    # Initialize the model and compute embeddings.
    model = SentenceTransformer(model_name)
    text_embeddings = model.encode(texts)
    name_embeddings = model.encode(names)
    curl_embeddings = model.encode(curls)
    # Attach embeddings (converted to lists for JSON serialization) to endpoints.
    for i, emb in enumerate(text_embeddings):
        endpoints[i]['text_embeddings'] = emb.tolist()
    for i, emb in enumerate(name_embeddings):
        endpoints[i]['name_embeddings'] = emb.tolist()
    for i, emb in enumerate(curl_embeddings):
        endpoints[i]['curl_embeddings'] = emb.tolist()

    # Save the results to the output file.
    with open(output_file, 'w') as f:
        json.dump(endpoints, f, indent=2)
    print(f"Embeddings for {len(endpoints)} endpoints have been saved to {output_file}")



def generate_curl_command(request):
    method = request.get("method", "GET").upper()
    cmd_parts = ["curl"]

    # Include method if not GET.
    if method != "GET":
        cmd_parts.extend(["-X", method])

    # Get URL (preferably the 'raw' key if available)
    url_obj = request.get("url")
    url = ""
    if isinstance(url_obj, dict):
        url = url_obj.get("raw", "")
    elif isinstance(url_obj, str):
        url = url_obj
    if url:
        cmd_parts.append(f'"{url}"')

    # Add headers if available.
    headers = request.get("header", [])
    for header in headers:
        key = header.get("key")
        value = header.get("value")
        if key and value:
            cmd_parts.append(f'-H "{key}: {value}"')

    # Add request body if available (only for raw data here).
    body_obj = request.get("body")
    if isinstance(body_obj, dict):
        mode = body_obj.get("mode")
        if mode == "raw" and "raw" in body_obj:
            raw_body = body_obj["raw"]
            cmd_parts.append(f"--data '{raw_body}'")

    return " ".join(cmd_parts)


def extract_endpoints(items, results):
    for item in items:
        if 'request' in item:
            name = item.get('name', '')
            description = item.get('description', '')
            request_obj = item.get('request', {})

            # Prefer description from the request object if available.
            if isinstance(request_obj, dict):
                description = request_obj.get('description', description)

            # Combine name and description.
            combined_text = f"Name: {name}. Description: {description}"

            # Generate a cURL command based on the request details.
            curl_command = generate_curl_command(request_obj)

            results.append({
                'name': name,
                'text': combined_text,
                'curl': curl_command
            })
        # If this item has nested items (e.g. folders), process them recursively.
        if 'item' in item:
            extract_endpoints(item['item'], results)
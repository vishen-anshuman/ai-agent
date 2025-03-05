import requests
# AIzaSyCGHxKeldWw19DtvlUcaoRj_ozLmQLxtOQ


def query_gemini(prompt: str, api_key = "AIzaSyCGHxKeldWw19DtvlUcaoRj_ozLmQLxtOQ"):
    """
    Queries Google's Gemini LLM API and returns the response.

    :param prompt: The text prompt to send to the LLM.
    :param api_key: Your Google AI API key.
    :return: The model's response as a string.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"

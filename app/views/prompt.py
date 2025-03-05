import json
import re

from rest_framework import status
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView


from .store.store import get_most_suitable_embedding
from .llm.gemini import query_gemini

class Prompt(APIView):
    renderer_classes = [JSONRenderer]
    def get(self, request):
        return Response({"message": "Hello, World!"}, status=status.HTTP_200_OK)

    def post(self, request):
        data = request.data
        support = "find the most suitable option for the prompt, also respond in JSON format as {\"index\" : integer_index}"
        print(data)
        value = get_most_suitable_embedding((data['prompt']))
        value_texts = []
        for text in value:
            value_texts.append(text['text'])
        res = query_gemini(prompt=f"{value} {support} {value_texts}")
        index = extract_index(res)
        return Response({"message": value_texts[index]}, status=status.HTTP_200_OK)


# python app.py migrate
# python app.py runserver
def check_top_similarity(top_5_results):
    """
    Checks if the top result has a similarity greater than 0.5.
    :param top_5_results: List of top 5 results with similarity scores.
    :return: Boolean value indicating whether the top result has similarity > 0.5.
    """
    if top_5_results and top_5_results[0]['similarity'] > 0.5:
        return True
    return False


def extract_index(data: str) -> int:
    try:
        if data is None:
            return -1
        data_str = data["candidates"][0]["content"]["parts"][0]["text"]
        clean_text = re.sub(r'```json\n|\n```', '', data_str).strip()
        json_data = json.loads(clean_text)
        return json_data["index"]
    except (KeyError, IndexError):
        return -1  # Return -1 or any other default value if extraction fails

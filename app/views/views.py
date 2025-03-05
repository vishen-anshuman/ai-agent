# from django.shortcuts import render
# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.renderers import JSONRenderer

class HelloWorld(APIView):
    renderer_classes = [JSONRenderer]
    def get(self, request):
        return Response({"message": "Hello, World!"}, status=status.HTTP_200_OK)

class GoodbyeWorld(APIView):
    renderer_classes = [JSONRenderer]
    def get(self, request):
        return Response({"message": "Goodbye, World!"}, status=status.HTTP_200_OK)
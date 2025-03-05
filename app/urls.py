
from django.urls import path
from .views import HelloWorld, GoodbyeWorld
from .views.prompt import Prompt

urlpatterns = [
    path('hello/', HelloWorld.as_view(), name='hello'),
    path('goodbye/', GoodbyeWorld.as_view(), name='goodbye'),
    path('prompt/', Prompt.as_view(), name='prompt'),
]
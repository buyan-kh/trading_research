from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Adjust according to your views
    # Add other paths as necessary
] 
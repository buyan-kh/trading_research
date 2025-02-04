from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Home page
    path('stock/', views.stock_view, name='stock_view'),  # Stock analysis page
    # Add other paths as necessary
] 
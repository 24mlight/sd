"""sd_multi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from simple import views, lb_views
from modules import ui

# single GPU machine
urlpatterns = [
    path('home/', views.homepage),
    # web tti
    path('multi_demo/', views.demo_page),
    path('txt2img/', views.txt2img),
    path('img2img/', views.img2img),
    path('progress/', views.progress),
    path('interrupt/', views.interrupt),
    path('list_models/', views.list_models),
]

# load balancing
# urlpatterns = [
#     path('txt2img/', lb_views.txt2img),
#     path('progress/', lb_views.progress),
#     path('interrupt/', lb_views.interrupt),
#     path('list_models/', lb_views.list_models),
# ]

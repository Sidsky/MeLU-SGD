from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path("about/", views.about, name='about'),
    path("generate-data", views.generateData, name="generate-data"),
    path("train", views.train_from_generate, name="redirect_to_train"),
]
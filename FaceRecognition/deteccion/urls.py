from django.urls import path
from . import views
from .views import principal_view , train_model , activar_camera , detect_people

urlpatterns = [
    #path("", views.index, name="index"),
    path('', principal_view, name='principal'),
    path('train_model/', train_model, name='train_model'),
    path('activar_camera/', activar_camera, name='activar_camera'),
    path('detect_people/', detect_people, name='detect_people'),
]
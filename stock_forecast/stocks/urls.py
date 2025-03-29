from django.urls import path
from .views import forecast_stock

urlpatterns = [
    path("forecast/", forecast_stock, name="forecast_stock"),

]

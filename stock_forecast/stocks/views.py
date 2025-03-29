from django.shortcuts import render

# Create your views here.
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from django.http import JsonResponse
from io import BytesIO
import base64

# Load Prophet models
def load_prophet_model(company_name):
    filename = f"stocks/models/prophet_model_{company_name}.pkl"
    model = joblib.load(filename)
    return model

# View for forecasting
def forecast_stock(request):
    if request.method == "GET":
        return render(request, "stocks/forecast.html")  # Form Page

    if request.method == "POST":
        company = request.POST.get("company")
        start_date = request.POST.get("start_date")
        period = int(request.POST.get("period"))
        freq = request.POST.get("frequency")

        # Load the appropriate model
        model = load_prophet_model(company)

        # Generate future dates
        future_dates = model.make_future_dataframe(periods=period, freq=freq)
        forecast = model.predict(future_dates)

        # Plot Forecast
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=forecast['ds'], y=forecast['yhat'], label="Predicted Price", color='blue')
        sns.lineplot(x=forecast['ds'], y=forecast['yhat_upper'], label="Upper Bound", linestyle='dashed', color='green')
        sns.lineplot(x=forecast['ds'], y=forecast['yhat_lower'], label="Lower Bound", linestyle='dashed', color='red')
        plt.title(f"Stock Price Forecast for {company}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()

        # Convert Plot to Image
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        image_base64 = base64.b64encode(image_png).decode("utf-8")

        return JsonResponse({"image": image_base64})

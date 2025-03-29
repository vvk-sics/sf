from django.shortcuts import render
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.http import JsonResponse
from io import BytesIO
import base64


# Function to load the Prophet model
def load_prophet_model(company_name):
    try:
        filename = f"stocks/models/prophet_model_{company_name}.pkl"
        model = joblib.load(filename)
        return model
    except Exception as e:
        return None

# Function to convert plot to base64 string
def plot_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode("utf-8")

# View for forecasting
def forecast_stock(request):
    if request.method == "GET":
        return render(request, "stocks/forecast.html")  # Render Form Page

    if request.method == "POST":
        company = request.POST.get("company")
        start_date = request.POST.get("start_date")
        period = request.POST.get("period")
        freq = request.POST.get("frequency")

        if not company or not start_date or not period:
            return JsonResponse({"error": "All fields are required."}, status=400)

        try:
            period = int(period)
        except ValueError:
            return JsonResponse({"error": "Period must be a number."}, status=400)

        model = load_prophet_model(company)
        if model is None:
            return JsonResponse({"error": f"Model for {company} not found."}, status=400)

        # Generate future dates and forecast
        future_dates = model.make_future_dataframe(periods=period, freq=freq)
        forecast = model.predict(future_dates)

        # Dictionary to store graphs
        graphs = {}

        # === 1. Forecast Plot ===
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=forecast['ds'], y=forecast['yhat'], label="Predicted Price", color='blue')
        sns.lineplot(x=forecast['ds'], y=forecast['yhat_upper'], label="Upper Bound", linestyle='dashed', color='green')
        sns.lineplot(x=forecast['ds'], y=forecast['yhat_lower'], label="Lower Bound", linestyle='dashed', color='red')
        plt.title(f"Stock Price Forecast for {company}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        graphs["forecast"] = plot_to_base64()
        plt.close()

        # === 2. Trend Plot ===
        plt.figure(figsize=(12, 4))
        sns.lineplot(x=forecast["ds"], y=forecast["trend"], label="Trend", color="purple")
        plt.title("Trend Component")
        plt.xlabel("Date")
        plt.ylabel("Trend Value")
        plt.legend()
        graphs["trend"] = plot_to_base64()
        plt.close()

        # === 3. Weekly Seasonality Plot ===
        plt.figure(figsize=(12, 4))
        sns.lineplot(x=forecast["ds"], y=forecast["weekly"], label="Weekly Seasonality", color="orange")
        plt.title("Weekly Seasonality Component")
        plt.xlabel("Date")
        plt.ylabel("Effect")
        plt.legend()
        graphs["weekly"] = plot_to_base64()
        plt.close()

        # === 4. Yearly Seasonality Plot ===
        plt.figure(figsize=(12, 4))
        sns.lineplot(x=forecast["ds"], y=forecast["yearly"], label="Yearly Seasonality", color="brown")
        plt.title("Yearly Seasonality Component")
        plt.xlabel("Date")
        plt.ylabel("Effect")
        plt.legend()
        graphs["yearly"] = plot_to_base64()
        plt.close()

        return JsonResponse(graphs)

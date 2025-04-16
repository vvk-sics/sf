from django.shortcuts import render
import joblib
import pandas as pd
import plotly.graph_objects as go
from django.http import JsonResponse
import io
import base64
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import plotly.graph_objects as go
from prophet.plot import plot_plotly
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def load_prophet_model(company_name):
    try:
        filename = f"stocks/models/prophet_model_{company_name}.pkl"
        model = joblib.load(filename)
        return model
    except Exception as e:
        return None


def forecast_stock(request):
    if request.method == "GET":
        return render(request, "stocks/forecast.html") 

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


        import datetime
        start_date_dt = pd.to_datetime(start_date)

        # Calculate how many days from the last training date to user-entered start date
        last_training_date = model.history['ds'].max()
        days_offset = (start_date_dt - last_training_date).days

        if days_offset > 0:
            total_period = days_offset + period
            future_dates = model.make_future_dataframe(periods=total_period, freq=freq)
            future_dates = future_dates[future_dates['ds'] >= start_date_dt]
        else:
            future_dates = model.make_future_dataframe(periods=period, freq=freq)
            future_dates = future_dates[future_dates['ds'] >= start_date_dt]        
        forecast = model.predict(future_dates)

      # === 1. Forecast Plot with Seasonality ===
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines+markers", name="Predicted Price", line=dict(color="blue")))
        fig1.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot", color="lightblue")))
        fig1.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot", color="lightblue")))
        fig1.add_trace(go.Scatter(x=forecast["ds"], y=forecast["weekly"], mode="lines", name="Weekly Seasonality", line=dict(color="orange")))
        fig1.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yearly"], mode="lines", name="Yearly Seasonality", line=dict(color="green")))
        fig1.update_layout(title=f"Stock Price Forecast for {company}", xaxis_title="Date", yaxis_title="Price", template="plotly_white")


        # === 2. Trend Plot ===
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["trend"], mode='lines', name="Trend", line=dict(color="purple")))
        fig2.update_layout(title="Trend Component", xaxis_title="Date", yaxis_title="Trend Value")

        # === 3. Weekly Seasonality Plot ===
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=forecast["ds"], y=forecast["weekly"], mode='lines', name="Weekly Seasonality", line=dict(color="orange")))
        fig3.update_layout(title="Weekly Seasonality Component", xaxis_title="Date", yaxis_title="Effect")

        # === 4. Yearly Seasonality Plot ===
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yearly"], mode='lines', name="Yearly Seasonality", line=dict(color="brown")))
        fig4.update_layout(title="Yearly Seasonality Component", xaxis_title="Date", yaxis_title="Effect")

        # === 5. Prophet Default Forecast Plot (Matplotlib rendered to base64) ===
        fig, ax = plt.subplots(figsize=(10, 5))
        model.plot(forecast, ax=ax)
        ax.set_title("Prophet Default Forecast Plot")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")

        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        plt.close(fig)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        image_url = f"data:image/png;base64,{image_base64}"

        return JsonResponse({
            "forecast": fig1.to_json(),
            "trend": fig2.to_json(),
            "weekly": fig3.to_json(),
            "yearly": fig4.to_json(),
            "prophet_default": image_url
        })

from django.db import models

# Create your models here.
from django.db import models

class ForecastRequest(models.Model):
    company = models.CharField(max_length=10, choices=[
        ('AAPL', 'Apple'),
        ('FB', 'Facebook'),
        ('AMD', 'AMD'),
        ('INTC', 'Intel')
    ])
    start_date = models.DateField()
    period = models.IntegerField()  # Number of days to forecast
    frequency = models.CharField(max_length=5, choices=[
        ('D', 'Daily'),
        ('W', 'Weekly'),
        ('M', 'Monthly')
    ])
    created_at = models.DateTimeField(auto_now_add=True)

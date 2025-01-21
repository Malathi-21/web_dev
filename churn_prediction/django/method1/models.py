from django.db import models

# Create your models here.
class churnModel(models.Model):

    tenure = models.IntegerField(null=True)
    MonthlyCharges=models.FloatField()
    TotalCharges=models.FloatField()
    OnlineSecurity_Yes=models.FloatField()
    Contract_Two_year=models.FloatField()

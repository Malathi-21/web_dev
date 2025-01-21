from django import forms
from .models import *


class churnForm(forms.ModelForm):
    class Meta():
        model=churnModel
        fields=['tenure','MonthlyCharges','TotalCharges','OnlineSecurity_Yes','Contract_Two_year']


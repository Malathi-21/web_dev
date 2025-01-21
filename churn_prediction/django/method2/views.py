from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
#from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc,classification_report,confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,chi2
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc

class dataUploadView(View):
    form_class = churnForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            data_tenure= request.POST.get('tenure')
            data_mc=request.POST.get('MonthlyCharges')
            data_tc=request.POST.get('TotalCharges')
            data_os=request.POST.get('OnlineSecurity_Yes')
            data_cy=request.POST.get('Contract_Two_year')
            #print (data)
            #dataset1=pd.read_csv("prep.csv",index_col=None)
            dicc={'yes':1,'no':0}

            df = pd.read_csv("prep.csv", index_col=None)
            x = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'OnlineSecurity_Yes', 'Contract_Two_year']]
            y = df["Churn_Yes"]

            # Standardize the selected features
            k = x.shape[1]  # Set k to the total number of columns
            selector = SelectKBest(score_func=chi2, k=k)
            X_selected = selector.fit_transform(x, y)
            
            scaler = StandardScaler()
            X_t = scaler.fit_transform(X_selected)

            # Define parameter grid and train the model
            param_grid = {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': [None, 'sqrt', 'log2'],
                'n_estimators': [10, 100]
            }
            grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3, n_jobs=-1)
            grid.fit(X_t, y)
            

            # Transform user input based on selected features
            data = np.array([[data_tenure,data_mc,data_tc,data_os,data_cy]])
            data_selected = scaler.transform(data)  
            data_scaled = scaler.transform(data_selected)
           
            out=grid.predict(data_scaled)

            return render(request, "succ_msg.html", {'data_tenure':data_tenure,'data_mc':data_mc,'data_tc':data_tc,'data_os':data_os,'data_cy':data_cy,
                                                        'out':out})
        
        else:
            return redirect(self.failure_url)

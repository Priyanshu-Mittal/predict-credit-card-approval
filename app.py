import streamlit as st
import webbrowser
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


st.title("Predicting Credit cards Approval")

if st.button('Source Code(Github)'):
    webbrowser.open_new_tab("https://github.com/Priyanshu-Mittal/predict-credit-card-approval")


st.header("**Data Exploration**")
cc_apps =pd.read_csv("datasets/cc_approvals.data",header=None)
st.subheader("Inspecting raw data:")
st.dataframe(cc_apps)

st.subheader("Data Description (Summary statistics):")
cc_apps_description = cc_apps.describe()
st.write(cc_apps_description)

st.header("**Data Preprocessing**")

st.markdown("Step 1: Replace the '?'s with NaN")
cc_apps = cc_apps.replace(to_replace='?',value=np.nan,inplace=False)
st.markdown("Step 2:Impute missing values in numeric columns using mean imputation ")
cc_apps.fillna(cc_apps.mean(),inplace=True)
st.markdown("Step 3:Impute missing values in categorical columns using most frequent values")
for col in cc_apps.columns:
    if cc_apps[col].dtype == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])



st.markdown("Step 4:Label Encoding to convert categorical values into numeric values")
le=LabelEncoder()
for col in cc_apps.columns.values:
    if cc_apps[col].dtypes =='object':
        cc_apps[col]=le.fit_transform(cc_apps[col])

st.markdown("Step 5:Feature selection: Drop non-useful columns(dropped DriversLicense,ZipCode)")
cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.values

st.markdown("Step 6:Split dataset in train and test set")
X,y = cc_apps[:,0:12] , cc_apps[:,13]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

st.markdown("Step 7:Data scaling")
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)


st.header("**Model Training and Prediction**")

st.markdown("Choose the model (Logistic Regression)")
logreg = LogisticRegression()
st.markdown("Model fitting")
logreg.fit(rescaledX_train,y_train)

st.markdown("Predicting the outputs")
y_pred = logreg.predict(rescaledX_test)

st.header("**Model Evaluation**")
st.write("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test,y_test))
st.write("Confusion matrix:",confusion_matrix(y_test,y_pred))


st.header("**Model Optimisation**")

st.markdown("Tuning the model hyperparameters(Used GridSearchCV)")
tol = [0.1,0.06,0.03,0.01,0.005,0.001,0.0001]
max_iter = [80,100,120,150,200]
param_grid = dict({'tol':tol, 'max_iter':max_iter})
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX, y)

best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_
st.write("Best Accuracy:" ,best_score)         



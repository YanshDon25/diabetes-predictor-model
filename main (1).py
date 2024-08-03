import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
data=pd.read_csv('diabetes.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.svm import SVC
cr=SVC(kernel='rbf',random_state=0)
cr.fit(x_train,y_train)
y_pred=cr.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)

def app():
  
  img=Image.open(r'diabetes.jpg.jpg')
  img=img.resize((100,100))
  st.image(img,width=200)
  st.title('Diabetes Prediction Model')
  st.sidebar.title('Diabetes Prediction')
  preg=st.sidebar.slider('Pregnancies',0,10,1)
  glu=st.sidebar.slider('Glucose',0.0,300.0,1.0)
  bp=st.sidebar.slider('BloodPressure',0.0,200.0,1.0)
  skin=st.sidebar.slider('SkinThickness',0.0,80.0,value=1.0)
  ins=st.sidebar.slider('Insulin',0.0,100.0,1.0)
  bmi=st.sidebar.slider('BMI',0.0,80.0,0.2)
  dpf=st.sidebar.slider('DiabetesPedigreeFunction',0.000,2.000,0.001)
  age=st.sidebar.slider('Age',0,80,1)

  input_data=np.array([preg,glu,bp,skin,ins,bmi,dpf,age]).reshape(1,-1)
  prediction=cr.predict(sc.transform(input_data))
  if prediction==0:
    st.write('The person is not diabetic')
  else:
    st.write('The person is diabetic')


if __name__=='__main__':
  app()


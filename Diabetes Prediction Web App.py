import numpy as np
import pickle
import streamlit as st
from sklearn.svm import SVC
#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()

SVC_model = pickle.load(open('Diabetes_prediction_model.sav','rb'))

def diabetes_prediction(input_data):
    #chnage the input data to numpy array
    input_data_numpy = np.asarray(input_data)

    #reshape the array as we are predicting for one row of data
    reshaped_data = input_data_numpy.reshape(1,-1)

    #standardize the data to be be in same format as our trained data
    #std_data = scaler.fit_transform(reshaped_data)
    
    #Make your prediction
    prediction = SVC_model.predict(reshaped_data)

    if prediction[0]==0:
        return 'Patient is not Diabetic'
    else:
        return 'Patient is Diabetic'

def main():
    
    #Giving a Title
    st.title('Diabetes Prediction Web App')
    
    #Getting the input Data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Values')
    SkinThickness = st.text_input('Skin Thickness Values')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Values')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Values')
    Age = st.text_input('Age of Patient')
    
    #code for prediction
    diagnosis =''
    
    #creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
                                        BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
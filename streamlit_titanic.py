import streamlit as st
import pickle as pl 
import pandas as pd
def titanic_app():
    with open('modelo_titanic.pkl', 'br') as file:
        ml = pl.load(file)
    with open('encoder.pkl', 'br') as file:
        encoder = pl.load(file)
    df = pd.read_csv('titanic.csv')
    sex = st.selectbox('Selecciona', list(df['Sex'].unique()))
    clase = st.selectbox('Selecciona', list(df['Pclass'].unique()))
    age = st.number_input('Selecciona la edad', min_value=0, max_value=110)
    fare = st.slider('Selecciona', min_value=df['Fare'].min(), max_value=df['Fare'].max())
    ciudad_salida = st.selectbox('Selecciona', list(df['Embark_Town'].unique()))


    #Aplicamos encodings
    dic_sex = { 'male':0, 'female': 1}
    sex_t = dic_sex[sex]
    dic_sex_transform = {0: 'male', 1: 'female'}
    embarked = list(encoder.transform([[ciudad_salida]]).toarray()[0])

    input_list = [[sex_t, age, clase, fare] + embarked]
    yhat = ml.predict(input_list)
    st.write(yhat)



    



    

    

if __name__ == '__main__':
    titanic_app()

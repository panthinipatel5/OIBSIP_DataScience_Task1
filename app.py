#import libraries
import pandas as pd #for file read and data manipulation
import numpy as np #for numeric calculation
import matplotlib.pyplot as plt #for data visualization
import seaborn as sns #for advance data visualization
import streamlit as st #for interactive dashboard
import joblib #for loading the model

#tab settings
st.set_page_config(
    page_title = 'Iris Flower Dashboard', 
    page_icon = '🌸', 
    layout = 'wide')

#load dataset
df = pd.read_csv('Iris.csv')

#creating a copy of dataset
df_copy = df.copy()

#removing unwanted columns
df_copy.drop('Id', axis = 1)

#load the model
model, le = joblib.load('iris_model.pkl')

#header

#dashboard title
st.title('🌸 Iris Flower Classification Dashboard')

#description
st.write("""
This dashboard predicts the species of an Iris Flower using Machine Learning
""")

#horizontal line
st.markdown('---')

#sidebar creation

#header
st.sidebar.header('Enter Flower Measurements')

#numer input fields
sepal_length = st.sidebar.number_input(
    'Sepal Length', 
    min_value = 0.0, 
    format = '%.2f')

sepal_width = st.sidebar.number_input(
    'Sepal Width',
    min_value = 0.0,
    format = '%.2f')

petal_length = st.sidebar.number_input(
    'Petal Length',
    min_value = 0.0,
    format = '%.2f')

petal_width = st.sidebar.number_input(
    'Petal Width',
    min_value = 0.0,
    format = '%.2f')

#predict button
if st.sidebar.button('Predict'):
    input_data = pd.DataFrame(
        [[sepal_length, 
          sepal_width, 
          petal_length,
          petal_width]],

        columns = [
            'SepalLengthCm',
            'SepalWidthCm',
            'PetalLengthCm',
            'PetalWidthCm'
        ]
    )

    prediction = model.predict(input_data)

    predicted_species = le.inverse_transform(prediction)

    st.success(f'Predicted Species: {predicted_species[0]}')

#main body

#first 5 rows
st.subheader('Dataset Preview')
st.dataframe(df.head())

#horizontal line
st.markdown('---')

#basic information about dataset
st.subheader('Dataset Information')

col1, col2 = st.columns(2)

with col1:
    st.write('Shape of Dataset')
    st.write(df.shape)

with col2:
    st.write('Missing Values in Dataset')
    st.write(df.isnull().sum())

#horizontal line
st.markdown('---')

#data visualization

fig1, ax1 = plt.subplots(figsize = (8,8))
sns.countplot(x = 'Species', data = df_copy, ax = ax1)

fig2, ax2 = plt.subplots(figsize = (8,8))
corr_matrix = df_copy.drop('Species', axis = 1).corr()
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', ax = ax2)

col3, col4 = st.columns(2)

with col3:
    st.subheader('Species Distribution')
    st.pyplot(fig1)

with col4:
    st.subheader('Correlation Heatmap')
    st.pyplot(fig2)

#horizontal line
st.markdown('---')

fig3, ax3 = plt.subplots(figsize = (8,8))
sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm',hue = 'Species', data = df_copy, ax = ax3)

fig4, ax4 = plt.subplots(figsize = (8,8))
sns.boxplot(x = 'Species', y = 'SepalLengthCm', data = df_copy, ax = ax4)

col5, col6 = st.columns(2)

with col5:
    st.subheader('Petal Length VS Petal Width')
    st.pyplot(fig3)

with col6:
    st.subheader('Sepal Length Distribution')
    st.pyplot(fig4)

#horizontal line
st.markdown('---')

st.subheader('Pair-Plot Visualization')
pair_plot = sns.pairplot(df_copy, hue = 'Species')
st.pyplot(pair_plot)

#horizontal line
st.markdown('---')

#footer
st.markdown('<h3 style = "text-align : center; color : #ffb6c1;">🌸 Made By Panthini Patel</h3>', unsafe_allow_html = True)
st.markdown('<p style = "text-align : center; font-size: 20px;">🚀 Made Using Streamlit</p>', unsafe_allow_html = True)

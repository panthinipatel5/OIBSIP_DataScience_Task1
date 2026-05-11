import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="Iris Flower Dashboard",
    page_icon="🌸",
    layout="wide"
)

df = pd.read_csv("Iris.csv")

df = df.drop("Id", axis=1)

model, le = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classification Dashboard")

st.write("""
This dashboard predicts the species of an Iris flower
using Machine Learning.
""")

st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.number_input(
    "Sepal Length",
    min_value=0.0,
    format="%.2f"
)

sepal_width = st.sidebar.number_input(
    "Sepal Width",
    min_value=0.0,
    format="%.2f"
)

petal_length = st.sidebar.number_input(
    "Petal Length",
    min_value=0.0,
    format="%.2f"
)

petal_width = st.sidebar.number_input(
    "Petal Width",
    min_value=0.0,
    format="%.2f"
)

if st.sidebar.button("Predict"):

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],

        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    )

    prediction = model.predict(input_data)

    predicted_species = le.inverse_transform(prediction)

    st.success(
        f"Predicted Species: {predicted_species[0]}"
    )

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.write("Shape of Dataset")
    st.write(df.shape)

with col2:
    st.write("Missing Values")
    st.write(df.isnull().sum())

st.subheader("Species Distribution")

fig1, ax1 = plt.subplots(figsize=(7,5))

sns.countplot(
    x="Species",
    data=df,
    ax=ax1
)

st.pyplot(fig1)

st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.heatmap(
    df.drop("Species", axis=1).corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)

st.subheader("Petal Length vs Petal Width")

fig3, ax3 = plt.subplots(figsize=(7,5))

sns.scatterplot(
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species",
    data=df,
    ax=ax3
)

st.pyplot(fig3)

st.subheader("Sepal Length Distribution")

fig4, ax4 = plt.subplots(figsize=(7,5))

sns.boxplot(
    x="Species",
    y="SepalLengthCm",
    data=df,
    ax=ax4
)

st.pyplot(fig4)

st.subheader("Pairplot Visualization")

pairplot_fig = sns.pairplot(
    df,
    hue="Species"
)

st.pyplot(pairplot_fig)

st.markdown("---")

st.write("Made using Streamlit")
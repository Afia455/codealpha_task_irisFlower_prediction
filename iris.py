import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64
import os


# Streamlit UI
st.title("🌺 Iris Flower Classification App")
st.write("Upload Iris Dataset and see details...")

# File uploader
# df = pd.read_csv("C:/Users/Home/Documents/GitHub/codealpha_task_irisFlower_prediction/Iris.csv")
df = pd.read_csv("Iris.csv")

if df is not None:
    df.columns = ['ID', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    # df.drop(columns=['Index', 'ID'], inplace=True) 
    
    # Display dataset preview
    st.subheader("📊 Dataset Preview")
    st.subheader("Header of dataset")
    st.write(df.head())
    st.subheader("End of Dataset")
    st.write(df.tail())
    

    st.subheader("EDA of Dataset: ")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Columns in Dataset")
    st.write(df.columns)

    st.subheader("Petal width and Length of Species")
    st.write(df.groupby('Species')[['PetalLengthCm', 'PetalWidthCm']].mean())

    
    st.subheader("Sepal width and Length of Species")
    st.write(df.groupby('Species')[['SepalLengthCm', 'SepalWidthCm']].mean())

    st.subheader("Iris Flower Species Name")
    st.write(df['Species'].unique())
    
    st.subheader("Visualization of Dataset: ")

    # Data Visualization
    st.subheader("Pairplot of Features")
    fig = sns.pairplot(df, hue='Species', diag_kind='kde')
    st.pyplot(fig)

    st.header("Division of species in Dataset")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Species", data=df, ax=ax)
    st.pyplot(fig)



    
    model = joblib.load("iris_model_down.pkl")
    label_encoder = joblib.load("label_encoder_down.pkl")
    scaler = joblib.load("scaler_down.pkl")


    def play_audio(mp3_file):
        audio_html = f"""
            <audio autoplay style="display: none;">
                <source src="data:audio/mp3;base64,{mp3_file}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)


# Load and encode MP3 file
    def get_audio_base64(file_path):
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode()



     
    st.header("🔍 Make a Prediction")
    sepal_length = st.number_input("🌿 Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.number_input("🌿 Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.number_input("🌸 Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.number_input("🌸 Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

    if st.button("🔮 Predict Species"):
         
         input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
         prediction = model.predict(input_data)
         predicted_species = label_encoder.inverse_transform(prediction)[0]

         

         st.success(f"🌼 Predicted Species: **{predicted_species}**")
         species = predicted_species[0]
         
         if predicted_species == "Iris-setosa":
                st.image("image/setosa.jpg", caption="Iris Setosa", use_container_width=True)
                st.write("This is Iris Setosa, a small and beautiful flower! 🌸")
                audio_base64 = get_audio_base64("voice.mp3")  
                st.balloons()

         elif predicted_species == "Iris-versicolor":
                st.image("image/versicolor.webp", caption="Iris Versicolor", use_container_width=True)
                st.write("This is Iris Versicolor, known for its vibrant colors! 🌿")
                audio_base64 = get_audio_base64("F:/jupyterProject/iris_classification/voice.mp3")                  
                st.audio("voice.mp3")
                st.balloons()

         elif predicted_species == "Iris-virginica":
                st.image("image/verginica.jpeg", caption="Iris Virginica", use_container_width=True)
                st.write("This is Iris Virginica, a tall and elegant flower! 🌺")
                audio_base64 = get_audio_base64("voice.mp3")  
                
         play_audio(audio_base64)      
         st.balloons()


























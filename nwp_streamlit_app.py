import streamlit as st
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    #st.write(tokenizer.word_index)

# Define the maximum sequence length
#as per data
max_sequence_length = 171

# Define a function to make predictions
def predict_next_word(text):
    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([text])
    
    # Pad the sequence
#1 less since during training time we used for input+target
    sequence = pad_sequences(sequence, maxlen=170, padding='pre')
    st.write(sequence)
    # Make a prediction using the model
    prediction = model.predict(sequence)
    
    # Get the index of the predicted word
    predicted_index = tf.argmax(prediction, axis=-1).numpy()[0]
    
    # Get the predicted word
    predicted_word = tokenizer.index_word[predicted_index]
    
    return predicted_word

# Create a Streamlit app
def main():
    st.title('🖥️ Next Word Prediction 📋')
    st.write('Enter text and get predictions for the next word(s).')
#     mydict={}
    # Add a text input widget
    input_text = st.text_input('Enter text:')
    # If the user has entered text
    if input_text:
#         try:
#             for i in range(1, len(input_text)):
#                 input_g= input_text[:i+1]
#                 f, l = input_g[:,:-1],input_g[:,-1]
#                 mydict[f]=l
#         st.write(mydict)    
        # Make a prediction for the next word
        
        prediction = predict_next_word(input_text)
        
        # Display the prediction
        st.write('Next word prediction:', prediction)
        
        # Add a button to add the prediction to the input text
        if st.button('Add prediction to input text'):
            input_text += ' ' + prediction
            st.write('Updated text:', input_text)
    st.write("Example: i am")
        
# Run the Streamlit app
if __name__ == '__main__':
    main()

#access it here https://next-word-prediction-app.streamlit.app/
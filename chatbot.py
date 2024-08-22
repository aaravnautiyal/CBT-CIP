import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
import keras

# Initialize lemmatizer to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Load the intents file and models
with open("C:/Users/super/Desktop/Visual/Personal_projects/python/python_chatbot/intents.json") as json_file:
    intent_data = json.load(json_file)

word_list = pickle.load(open("words.pkl", "rb"))
intent_labels = pickle.load(open("classes.pkl", "rb"))
chat_model = keras.models.load_model("chatbot_model.h5")


# Function to clean up and tokenize input sentence
def preprocess_input(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens

# Create a bag of words representation of the input sentence
def vectorize_sentence(sentence):
    tokens = preprocess_input(sentence)
    vector = [0] * len(word_list)
    for token in tokens:
        for index, word in enumerate(word_list):
            if word == token:
                vector[index] = 1
    return np.array(vector)

# Predict the class label for the input sentence
def classify_intent(sentence):
    vector = vectorize_sentence(sentence)
    predictions = chat_model.predict(np.array([vector]))[0]
    threshold = 0.25
    filtered_predictions = [(i, prob) for i, prob in enumerate(predictions) if prob > threshold]

    # Sort by highest probability
    filtered_predictions.sort(key=lambda x: x[1], reverse=True)
    result = [{"intent": intent_labels[i], "probability": str(prob)} for i, prob in filtered_predictions]
    return result

# Generate a response based on the predicted intent
def fetch_response(predicted_intent, intents_json):
    intent_tag = predicted_intent[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])

print("Chatbot is now active and ready to chat!")

# Main loop for continuous user interaction
while True:
    user_input = input("You: ")
    predicted_intents = classify_intent(user_input)
    chatbot_response = fetch_response(predicted_intents, intent_data)
    print(f"Bot: {chatbot_response}")

import random
import json
import pickle
import numpy as np
import tensorflow as tf
import sys
import io

import nltk
from nltk.stem import WordNetLemmatizer

# Redirecting standard output and error to handle UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Initialize lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open("C:/Users/super/Desktop/Visual/Personal_projects/python/python_chatbot/intents.json", encoding='utf-8') as file:
    intents_data = json.load(file)

# Initialize lists for words, classes, and documents
all_words = []
tags = []
data_pairs = []
exclude_chars = ["?", "!", ".", ","]

# Process each intent and its patterns
for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        tokenized_words = nltk.word_tokenize(pattern)
        all_words.extend(tokenized_words)
        data_pairs.append((tokenized_words, intent["tag"]))
        if intent["tag"] not in tags:
            tags.append(intent["tag"])

# Lemmatize and sort the words and classes
all_words = sorted(set(lemmatizer.lemmatize(word.lower()) for word in all_words if word not in exclude_chars))
tags = sorted(set(tags))

# Save words and tags to disk
with open("words.pkl", "wb") as words_file:
    pickle.dump(all_words, words_file)

with open("classes.pkl", "wb") as classes_file:
    pickle.dump(tags, classes_file)

# Prepare training data
training_data = []
empty_output = [0] * len(tags)

# Print the processed documents for verification
print("Processed Documents:\n", data_pairs)

for words, tag in data_pairs:
    bag_vector = []
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    for word in all_words:
        bag_vector.append(1) if word in words else bag_vector.append(0)

    output_vector = list(empty_output)
    output_vector[tags.index(tag)] = 1
    training_data.append(bag_vector + output_vector)

# Shuffle and convert training data to numpy array
random.shuffle(training_data)
training_data = np.array(training_data)

# Print the training examples for verification
print('\nTraining Examples:\n', training_data)

# Split the data into input (X) and output (Y)
train_X = training_data[:, :len(all_words)]
train_Y = training_data[:, len(all_words):]

# Print the shape of training data
print("Training X Shape:", train_X.shape)
print("Training Y Shape:", train_Y.shape)

# Suppress TensorFlow warnings and information logs
tf.get_logger().setLevel('ERROR')

# Define the neural network model
chatbot_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_X[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_Y[0]), activation='softmax')
])

# Compile the model with SGD optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
chatbot_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model, handling any potential errors
try:
    training_history = chatbot_model.fit(np.array(train_X), np.array(train_Y), epochs=200, batch_size=5, verbose=1)
except Exception as error:
    print("Error during model training:", error)

# Save the trained model to disk
chatbot_model.save("chatbot_model.h5")
print("Model training complete and saved.")

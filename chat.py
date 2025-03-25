import random
import torch
import json
import nltk
from train import ChatBotModel, all_words, tags  # Ensure to import your model and utils

# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

# Load the model
input_size = len(all_words)  # Adjust to the actual size of your feature vector
hidden_size = 16  # Same as defined in train.py
output_size = len(tags)

model = ChatBotModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('chatbot_model.pth'))
model.eval()

def bag_of_words(msg, all_words):
    # Tokenize the input message
    tokenized_message = nltk.word_tokenize(msg)
    # Create a bag of words representation
    return [1 if w in tokenized_message else 0 for w in all_words]

def chatbot_response(msg):
    # Process user message
    input_bag = bag_of_words(msg, all_words)
    input_tensor = torch.tensor(input_bag, dtype=torch.float32)

    # Predict tag
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, dim=0)
    tag = tags[predicted.item()]

    # Get response based on the predicted tag
    response = get_response(tag)
    return response, tag

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I don't understand."

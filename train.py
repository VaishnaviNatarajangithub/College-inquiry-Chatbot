'''import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')

# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Prepare the data
all_words = []
tags = []
ignore_words = ['?', '!', '.', ',', "'s", "'m", "'ll", "'ve", "'d", "'re", "'t", "is", "are", "am"]  # Define ignore words

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        all_words.extend(word_list)
        # Add the tag to the tags list
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Lemmatize and filter out the ignore words
all_words = [lemmatizer.lemmatize(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))  # Remove duplicates and sort

# Encode tags
label_encoder = LabelEncoder()
tags = sorted(set(tags))
labels = label_encoder.fit_transform(tags)

# Create bag of words function
def bag_of_words(msg, all_words):
    """Creates a bag of words representation for the given message."""
    msg_words = nltk.word_tokenize(msg)
    msg_words = [lemmatizer.lemmatize(word.lower()) for word in msg_words]
    bag = [0] * len(all_words)  # Initialize bag
    for w in msg_words:
        for i, word in enumerate(all_words):
            if word == w:
                bag[i] = 1  # Mark the word as present
    return np.array(bag)

# Create training data
training = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Create the bag of words
        bag = [0] * len(all_words)  # Initialize the bag
        word_list = nltk.word_tokenize(pattern)
        for w in word_list:
            for i, word in enumerate(all_words):
                if word == w.lower():
                    bag[i] = 1  # Mark the word as present
        # Ensure the shape is consistent before adding to training
        label = label_encoder.transform([intent['tag']])[0]
        training.append((bag, label))

# Shuffle the training data
np.random.shuffle(training)

# Split into inputs and outputs
X_train = np.array([item[0] for item in training])  # Bag of words
y_train = np.array([item[1] for item in training])  # Labels

# Define the model
class ChatBotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training parameters
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)
num_epochs = 1000
learning_rate = 0.001

# Initialize model, loss function, and optimizer
model = ChatBotModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Convert to tensors
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'chatbot_model.pth')
print("Training complete. Model saved as 'chatbot_model.pth'") '''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('wordnet')

# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Prepare the data
all_words = []
tags = []
ignore_words = ['?', '!', '.', ',', "'s", "'m", "'ll", "'ve", "'d", "'re", "'t", "is", "are", "am"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        all_words.extend(word_list)
        # Add the tag to the tags list
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Lemmatize and filter out the ignore words
all_words = [lemmatizer.lemmatize(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))

# Encode tags
label_encoder = LabelEncoder()
tags = sorted(set(tags))
labels = label_encoder.fit_transform(tags)

# Create bag of words function
def bag_of_words(msg, all_words):
    """Creates a bag of words representation for the given message."""
    msg_words = nltk.word_tokenize(msg)
    msg_words = [lemmatizer.lemmatize(word.lower()) for word in msg_words]
    bag = [0] * len(all_words)
    for w in msg_words:
        for i, word in enumerate(all_words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Create training data
training = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Create the bag of words
        bag = bag_of_words(pattern, all_words)
        label = label_encoder.transform([intent['tag']])[0]
        training.append((bag, label))

# Shuffle the training data
np.random.shuffle(training)

# Split into inputs and outputs
X_train = np.array([item[0] for item in training])  # Bag of words
y_train = np.array([item[1] for item in training])  # Labels

# Define the model
class ChatBotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training parameters
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)
num_epochs = 1000
learning_rate = 0.001

# Initialize model, loss function, and optimizer
model = ChatBotModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Convert to tensors
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'chatbot_model.pth')
print("Training complete. Model saved as 'chatbot_model.pth'")

# ==========================================
# Accuracy Check
# ==========================================
# Model evaluation
model.eval()
with torch.no_grad():
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_train, predicted.numpy())
    print(f"Accuracy: {accuracy * 100:.2f}%")



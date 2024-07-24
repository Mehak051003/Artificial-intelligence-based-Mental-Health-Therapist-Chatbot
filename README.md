# AI-Based Mental Health Therapist Chatbot
# Overview
This project involves the development of an AI-based mental health therapist chatbot using a pre-trained seq2seq model with LSTM layers and a Flask web application for user interaction.

# Tools and Requirements
## Development Environment
Visual Studio Code: Recommended for development.

## Dependencies
Install the required Python libraries by running:

`pip install flask keras numpy difflib`

## Files Needed
1. training_model.h5: Pre-trained neural network model.
2. knowledge_base.json: Knowledge base for the chatbot.
3. Feature dictionaries: input_features_dict, target_features_dict, reverse_target_features_dict.

# Methodology
1. Importing Libraries
   
`import json
import numpy as np
from difflib import get_close_matches
from keras.models import load_model, Model
from keras.layers import Input, LSTM, Dense
from flask import Flask, render_template, request
import re
import os`

2. Initializing Flask App

`app = Flask(__name__, template_folder='templates', static_folder='static')`

3. ChatBot Class Initialization

class ChatBot:

    def __init__(self):
        self.training_model = load_model('training_model.h5')
        encoder_inputs = self.training_model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = self.training_model.layers[2].output
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)
        self.knowledge_base = self.load_knowledge_base('knowledge_base.json')
        self.question_dict = self.build_question_dict(self.knowledge_base["questions"])

    def find_best_match(self, user_question: str) -> str | None:
        matches = get_close_matches(user_question, self.question_dict.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None

    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = self.decode_response(input_matrix)
        chatbot_response = chatbot_response.replace("<START>", '').replace("<END>", '')
        return chatbot_response

    def render_chat(self, messages):
        return render_template('chat.html', messages=messages)

    def run_web_chat_bot(self):
        messages = []
        @app.route('/')
        def index():
            return self.render_chat(messages)
        @app.route('/send', methods=['POST'])
        def send():
            user_input = request.form['user_input']
            bot_response = self.generate_response(user_input)
            messages.append({'sender': 'user', 'message': user_input})
            messages.append({'sender': 'bot', 'message': bot_response})
            return self.render_chat(messages)
         
         if __name__ == '__main__':
             ch = ChatBot()
             ch.run_web_chat_bot()`
             
4. HTML Template (chat.html)

         `<div id="chat-container">
             {% for message in messages %}
                 {% if message.sender == 'bot' %}
                     <p class="bot-message">Bot: {{ message.message }}</p>
                 {% elif message.sender == 'user' %}
                     <p class="user-message">User: {{ message.message }}</p>
                 {% endif %}
             {% endfor %}
         </div>`

# Result and Discussion
Model Loading and Architecture: Successfully loads a pre-trained seq2seq model.

Knowledge Base: Efficient handling of user queries and dynamic updates.

Fuzzy Matching: Improved handling of variations in user queries.

Interactive Chat Loop: Engages users in a dynamic conversation.

Web Interface (Flask): Real-time chat messages and user-friendly interface.

![image](https://github.com/user-attachments/assets/9ee8b069-d32a-4aed-98d3-fc5011d10c22)    ![image](https://github.com/user-attachments/assets/01e127f5-9a58-4e5c-bd62-97e1ea3477e4)


# Vision
Create a sophisticated conversational agent that provides personalized, empathetic, and contextually relevant interactions, anticipating and exceeding future user requirements.

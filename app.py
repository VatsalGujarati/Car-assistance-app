#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template_string
from flask_ngrok import run_with_ngrok
import transformers
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pyttsx3
import logging
import re
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and tokenizers
gpt2_model_name = "gpt2-medium"
translation_model_name = "Helsinki-NLP/opus-mt-en-de"

gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = transformers.AutoModelForCausalLM.from_pretrained(gpt2_model_name)
translator_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# NLU pipeline for named entity recognition
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Simulated car state
car_state = {
    "temperature": 22,
    "music_playing": False,
    "navigation_active": False,
    "windows": "closed",
    "sunroof": "closed",
    "location": "Unknown",
}

# User profile for preferences
user_profile = {"name": "User", "preferences": {"music": "rock", "language": "en"}}

def generate_response(text):
    input_ids = gpt2_tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    attention_mask = torch.ones_like(input_ids)
    gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id
    output = gpt2_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    response = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def process_command(text):
    logger.info(f"Processing command: {text}")

    text = text.lower()
    if 'play music' in text:
        return play_music()
    elif 'navigate to' in text:
        destination = re.sub(r'navigate to', '', text).strip()
        return navigate_to(destination)
    elif 'set temperature' in text:
        temperature = re.search(r'\d+', text)
        if temperature:
            return set_temperature(int(temperature.group()))
    elif 'open' in text or 'close' in text:
        return control_windows_or_sunroof(text)
    else:
        return "I'm not sure how to help with that."

def play_music():
    global car_state
    car_state["music_playing"] = True
    genre = user_profile['preferences']['music']
    response = f"Playing {genre} music."
    logger.info(response)
    return response

def navigate_to(destination):
    global car_state
    car_state["navigation_active"] = True
    car_state["location"] = destination
    response = f"Starting navigation to {destination}."
    logger.info(response)
    return response

def set_temperature(temperature):
    global car_state
    car_state["temperature"] = temperature
    response = f"Setting temperature to {temperature} degrees Celsius."
    logger.info(response)
    return response

def control_windows_or_sunroof(text):
    global car_state
    if 'open' in text:
        if 'window' in text:
            car_state['windows'] = 'open'
        if 'sunroof' in text:
            car_state['sunroof'] = 'open'
    elif 'close' in text:
        if 'window' in text:
            car_state['windows'] = 'closed'
        if 'sunroof' in text:
            car_state['sunroof'] = 'closed'
    response = f"Windows are {car_state['windows']} and sunroof is {car_state['sunroof']}."
    logger.info(response)
    return response

# Flask app setup
app = Flask(__name__)
run_with_ngrok(app)  # Integrate Flask with ngrok

@app.route('/')
def home():
    return render_template_string("""
    <!doctype html>
    <html>
        <head>
            <title>Car Voice Assistant</title>
            <script>
                function startRecognition() {
                    var recognition = new webkitSpeechRecognition();
                    recognition.lang = "en-US";
                    recognition.onresult = function(event) {
                        document.getElementById('command').value = event.results[0][0].transcript;
                        document.getElementById('commandForm').submit();
                    }
                    recognition.start();
                }
                function speak(text) {
                    var synth = window.speechSynthesis;
                    var utterance = new SpeechSynthesisUtterance(text);
                    synth.speak(utterance);
                }
            </script>
        </head>
        <body>
            <h1>Test the Car Voice Assistant</h1>
            <p>Press the button and speak a command. Example commands:</p>
            <ul>
                <li>"Play music"</li>
                <li>"Navigate to Berlin"</li>
                <li>"Set temperature to 24 degrees"</li>
                <li>"Open the windows"</li>
            </ul>
            <form id="commandForm" action="/process" method="post">
                <input type="hidden" id="command" name="command">
            </form>
            <button onclick="startRecognition()">Press and Speak</button>
            <p id="response">{{response}}</p>
            <h2>Background Process</h2>
            <pre>{{background}}</pre>
            <script>
                var response = "{{response}}";
                if (response) {
                    speak(response);
                }
            </script>
        </body>
    </html>
    """, response="", background="Waiting for command...")

@app.route('/process', methods=['POST'])
def process():
    command = request.form['command']
    logger.info(f"User said: {command}")
    response = process_command(command)
    background = f"Processed command: {command}\nResponse: {response}\n\nCurrent Car State: {car_state}"
    logger.info(f"Response: {response}")
    return render_template_string("""
    <!doctype html>
    <html>
        <head>
            <title>Car Voice Assistant</title>
            <script>
                function startRecognition() {
                    var recognition = new webkitSpeechRecognition();
                    recognition.lang = "en-US";
                    recognition.onresult = function(event) {
                        document.getElementById('command').value = event.results[0][0].transcript;
                        document.getElementById('commandForm').submit();
                    }
                    recognition.start();
                }
                function speak(text) {
                    var synth = window.speechSynthesis;
                    var utterance = new SpeechSynthesisUtterance(text);
                    synth.speak(utterance);
                }
            </script>
        </head>
        <body>
            <h1>Test the Car Voice Assistant</h1>
            <p>Press the button and speak a command. Example commands:</p>
            <ul>
                <li>"Play music"</li>
                <li>"Navigate to Berlin"</li>
                <li>"Set temperature to 24 degrees"</li>
                <li>"Open the windows"</li>
            </ul>
            <form id="commandForm" action="/process" method="post">
                <input type="hidden" id="command" name="command">
            </form>
            <button onclick="startRecognition()">Press and Speak</button>
            <p id="response">Response: {{response}}</p>
            <h2>Background Process</h2>
            <pre>{{background}}</pre>
            <script>
                var response = "{{response}}";
                if (response) {
                    speak(response);
                }
            </script>
        </body>
    </html>
    """, response=response, background=background)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


# In[ ]:





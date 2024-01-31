from __future__ import division, print_function
from flask_mysqldb import MySQL

from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template, jsonify, Response
import pymysql
import cv2
import pickle
import imutils
from nltk.stem import WordNetLemmatizer
import json
import random
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import nltk
lemmatizer = WordNetLemmatizer()

#
from flask import Flask, render_template, Response, request
from chatbot import chatbot_response

from deuteranomali import generate_frames
from deuteranopia import generate_frames

from tensorflow.keras.models import load_model
from flask_mysqldb import MySQL
from flask import Flask, render_template, Response, request




from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("PartialHelper.html", page="home")


# Model saved with Keras model.save()
MODEL_PATH = 'model/chatbot_model.h5'

# Load your trained model
model= load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')

# model_chatbot = load_model('model/chatbot_model.h5')
# intents = json.loads(open('model/intents.json').read())
# words = pickle.load(open('model/words.pkl', 'rb'))
# classes = pickle.load(open('model/classes.pkl', 'rb'))

model_chat = load_model('model/chatbot_model.h5')
with open('model/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
    
    
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))

def predict_class(sentence, model_chat):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model_chat.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model_chat)
    res = getResponse(ints, intents)
    return res

@app.route('/chatbot', methods=['GET'])
def chatbot():
    return render_template('chatbot.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/ishihara')
def index():
    return render_template('ishihara.html')





# #upload gambar
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/uploads')
# def index():
#     return render_template('upload.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return redirect(request.url)

#     file = request.files['image']

#     if file.filename == '':
#         return redirect(request.url)

#     if file and allowed_file(file.filename):
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filename)
#         return 'Upload successful!'

#     return 'Invalid file type.'


##







    
    

@app.route("/deuteranomali")
def deuteranomali():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/deuteranopia")
def deuteranopia():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/protanomali")
def protanomali():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/protanopia")
def protanopia():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/tritanomali")
def tritanomali():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/tritanopia")
def tritanopia():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )





    





















































































from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import pickle
from nltk.stem import WordNetLemmatizer
import json
import random
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import nltk
lemmatizer = WordNetLemmatizer()
from flask_mysqldb import MySQL

# Keras

# Flask utils

# Define a flask app

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'capstone'

mysql = MySQL(app)



# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')



       

# Load SVM model
with open('model/model_svm.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load TfidfVectorizer
with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# def preprocess_text(text):
#     # Preprocess the text (e.g., remove special characters, lowercase)
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = text.lower()
#     return text




@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# @app.route('/ishihara')
# def feedback():
#     return render_template('ishihara.html')

# @app.route('/sentimen', methods=['POST'])
# def sentimen():
#     if request.method == 'POST':
#         feedback_text = request.form['feedback']
#         # Preprocess the input text
#         preprocessed_text = preprocess_text(feedback_text)
#         # Transform the input text using the TfidfVectorizer
#         input_vector = tfidf_vectorizer.transform([preprocessed_text]).toarray()
#         # Make prediction using the SVM model
#         prediction = svm_model.predict(input_vector)[0]
#         print("Prediction:", prediction)
        
#         sentiment_mapping = {0: 'Negatif', 1: 'Positif'}
#         sentiment = sentiment_mapping[prediction]
 
#         return render_template('feedback.html', feedback=feedback_text, sentiment=sentiment)

@app.route('/sentimen', methods=['POST'])
def sentimen():
    if request.method == 'POST':
        feedback = request.form['feedback']

        # Lakukan analisis sentimen
        sentiment = predict_sentiment(feedback)

        # Simpan ke database
        save_to_database(feedback, sentiment)

         # Hitung persentase hasil sentimen
        positive_percentage, negative_percentage = calculate_sentiment_percentage()
        
        # Ambil data dari database untuk ditampilkan di HTML
        cur = mysql.connection.cursor()
        cur.execute("SELECT feedback_user, hasil_sentimen_analisis FROM feedback ORDER BY id DESC LIMIT 1")
        result = cur.fetchone()
        cur.close()

        user_input = result[0]
        sentiment_result = result[1]

        return render_template('feedback.html', user_input=user_input, sentiment_result=sentiment_result, positive_percentage=positive_percentage, negative_percentage=negative_percentage)
    return render_template('feedback.html')

def predict_sentiment(feedback):
    # Lakukan pra-pemrosesan pada feedback (sesuai dengan langkah preprocessing)
    transformed_feedback = tfidf_vectorizer.transform([feedback])

    # Lakukan prediksi sentimen menggunakan model
    prediction = svm_model.predict(transformed_feedback)

    if prediction[0] in [0]:
        return 'negatif'
    elif prediction[0] in [1]:
        return 'positif'

def save_to_database(feedback, sentiment):
    # Simpan data ke database
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO feedback (feedback_user, hasil_sentimen_analisis) VALUES (%s, %s)", (feedback, sentiment))
    mysql.connection.commit()
    cur.close()

def calculate_sentiment_percentage():
    # Ambil jumlah feedback untuk masing-masing sentimen
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback WHERE hasil_sentimen_analisis='positif'")
    positive_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM feedback WHERE hasil_sentimen_analisis='negatif'")
    negative_count = cur.fetchone()[0]

    total_count = positive_count + negative_count

    # Hitung persentase
    positive_percentage = (positive_count / total_count) * 100

    negative_percentage = (negative_count / total_count) * 100

    return positive_percentage, negative_percentage





if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')


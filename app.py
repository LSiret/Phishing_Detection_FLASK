from flask import Flask, render_template_string, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from email_validator import validate_email, EmailNotValidError
import os
import subprocess
import requests
import time
import uuid
import re
import joblib
import PyPDF2
import sqlite3

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": "Bearer hf_XXXXXXXXXXXXXXXXX"}

scam_email_model = joblib.load("data/scam_email_model.joblib")
scam_types_model = joblib.load("data/scam_types_model.joblib")
vectorizer = joblib.load("data/vectorizer.joblib")

# LLM
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Program functions

# Read all text on a PDF
def pdf_to_text(pdf_path):
    # Extract text from PDF using PyPDF2
    pdf = PyPDF2.PdfReader(pdf_path)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Process the text before using a model
def process_text(text):
    # Remove hyperlinks
    text = re.sub(r'http\S+', '', text)

    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenise the text
    tokenized_text = vectorizer.transform([text])

    return tokenized_text

# Main binary prediction
def predict_text(text):
    tokenized_text = process_text(text)

    # Make prediction
    prediction = scam_email_model.predict(tokenized_text)
    return prediction

# Convert binary prediction to name
def prediction_to_text(prediction):
    if prediction > 0.5:
        prediction = "Legitimate"
    else:
        prediction = "Scam"
    return prediction

# Predict the category of scam
def predict_type(text):
    tokenized_text = process_text(text)

    # Make prediction
    prediction = scam_types_model.predict(tokenized_text)
    prediction = prediction.argmax(axis=0) + 1  # Add 1 to match class labels (1-4)
    return prediction

# Convert type number to name
def type_to_text(prediction):
    prediction = round(prediction)
    if prediction == 1:
        prediction = "Commercial Spam"
    elif prediction == 2:
        prediction = "False Positives"
    elif prediction == 3:
        prediction = "Fraud"
    elif prediction == 4:
        prediction = "Phishing"
    return prediction

# Validate password
def validate_password(password):
    # Require at least 8 characters, one uppercase letter, one lowercase letter, one digit, and one special character
    if (len(password) >= 8 and
        re.search(r"[A-Z]", password) and
        re.search(r"[a-z]", password) and
        re.search(r"[0-9]", password) and
        re.search(r"[\W_]", password)):
        return True
    return False


# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Set a secret key for session management
app.secret_key = os.urandom(24)

# Database initialization function
def init_db():
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

# Initialise database
init_db()

# Set upload folder inside the static directory
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# HTML templates
register_template = open("templates/register.html", 'r').read()
login_template = open("templates/login.html", 'r').read()
main_template = open("templates/index.html", 'r').read()


# Route for registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate email format
        try:
            valid = validate_email(email)
            email = valid.email
        except EmailNotValidError as e:
            return render_template_string(register_template, error=str(e))

        # Validate password strength
        if not validate_password(password):
            return render_template_string(register_template, error='Password must be at least 8 characters long, contain an uppercase letter, a lowercase letter, a digit, and a special character.')

        # Hash the password before storing
        hashed_password = generate_password_hash(password)

        # Store user in the database
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
                conn.commit()
            except sqlite3.IntegrityError:
                return render_template_string(register_template, error='Email is already registered.')

        return redirect(url_for('login'))

    return render_template_string(register_template)

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if the email is registered
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()

        if row and check_password_hash(row[0], password):
            session['email'] = email  # Store email in session
            return redirect(url_for('index'))
        else:
            return render_template_string(login_template, error='Invalid email or password.')

    return render_template_string(login_template)

# Route for logout
@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    
    # Redirect users not logged in
    if 'email' not in session:
        return redirect(url_for('login'))
    
    # Store chat data
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    if 'latest_prediction' not in session:
        session['latest_prediction'] = None

    prediction = None
    advice = None

    if request.method == 'GET':
        text = None
        if request.args.get('body'):
            text = ''
            text += str(request.args.get('sender'))
            text += str(request.args.get('subject'))
            text += str(request.args.get('body'))
        if text:
            # Run predictions
            prediction = predict_text(text)
            prediction = prediction_to_text(prediction)

    elif request.method == 'POST':        
        
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Generate a unique filename using UUID
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
    
            # Run predictions
            text = pdf_to_text(file_path)
            prediction = predict_text(text)
            prediction = prediction_to_text(prediction)

            # Store the uploaded file name for display
            uploaded_file = unique_filename
            

    if prediction == "Scam":
        prediction = predict_type(text)
        prediction = type_to_text(prediction)
        session['latest_prediction'] = prediction
        match prediction:
            case "Commercial Spam":
                advice = "These types of emails are unlikely to do any harm. You may delete it to free up your inbox, or you are receiving these frequently, there may be an option to unsubscribe from their email list."
            case "False Positive":
                advice = "Our system is having difficulties classifying this email. It is likely that it is spam, however it would be worth checking if you were expecting this email."
            case "Fraud":
                advice = "Delete this email and similar emails in future. You should ignore emails like these and be extremely cautious if someone is asking for and of your information."
            case "Phishing":
                advice = "Delete this email and similar emails in future. You should ignore emails like these and be extremely cautious if someone is asking for and of your information."
    else:
        advice = "This email appears to be legitimate although always take caution when clicking on any links in an email. Never send any personal data over email unless you know who you are sending it to."
    
    # Process LLM chat separately
    if request.method == 'GET':
        
        # Get user input
        user_input = request.args.get('user_input', '').strip()

        # Handle user input
        if user_input:
            prompt = {"inputs": f"Imagine an email that is {session['latest_prediction']}. Answer the user's query: {user_input}"}
            response = query(prompt)
            ai_response = response[0].get('generated_text', "Sorry, I couldn't generate a proper response.")
            
            session['conversation_history'].append({'user': user_input, 'ai': ai_response})


    return render_template_string(main_template, result=prediction, advice=advice, conversation_history=session['conversation_history'])


# Main entry point of the application, this block runs when the script is executed directly
if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run()


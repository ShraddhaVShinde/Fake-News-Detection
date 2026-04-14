from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask
app = Flask(__name__)

# Load the trained Model and Vectorizer saved from your Jupyter Notebook
model = pickle.load(open('fake_news_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# NLP Setup
nltk.download('stopwords')
ps = PorterStemmer()

def preprocess(text):
    # Standard cleaning: regex, lowercase, stopword removal, stemming
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

@app.route('/')
def home():
    # Renders the initial empty form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Get user input from the form
        raw_text = request.form['news_text']
        
        if not raw_text.strip():
            return render_template('index.html', prediction_text="Please enter text!")

        # 2. Apply Modeling Pipeline (Clean -> Transform -> Predict)
        cleaned_text = preprocess(raw_text)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        
        # 3. Map numerical output to labels (0=Fake, 1=Real)
        result = "FAKE" if prediction[0] == 0 else "REAL"
        
        # 4. Return result back to the UI
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
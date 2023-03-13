from flask import Flask, jsonify , request
import numpy as np
import pandas as pd
import pickle
import spacy
import sentence_transformers

app = Flask(__name__)

logreg = pickle.load(open('/content/drive/MyDrive/logreg.pkl', 'rb'))

nlp = spacy.load('/content/drive/MyDrive/spacy_nlp_model/content/spacy_nlp_model')

model = sentence_transformers.SentenceTransformer('/content/drive/MyDrive/saved_transformer_model')

@app.post('/predict')
def predict():
  data = request.json['text']
  print(data)
  return None


if __name__=='__main__':
    app.run('0.0.0.0',debug=True)
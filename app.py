from flask import Flask, jsonify , request
import string
import numpy as np
import pandas as pd
import pickle
import spacy
import sentence_transformers

app = Flask(__name__)

# Loading all trained models

logreg = pickle.load(open('/content/drive/MyDrive/logreg.pkl', 'rb'))

nlp = spacy.load('/content/drive/MyDrive/spacy_nlp_model/content/spacy_nlp_model')

model = sentence_transformers.SentenceTransformer('/content/drive/MyDrive/saved_transformer_model')


@app.post('/predict')
def predict():

  data = request.json['text']

  sentence_1 , sentence_2 = data.split('#')[0] , data.split('#')[1]
  
  file_lines=[]

  file_lines.append(sentence_1)
  file_lines.append(sentence_2)

  df = pd.DataFrame(file_lines).T

  df.columns =['sentence_1','sentence_2']

  df['sentence_1'] = df['sentence_1'].apply(lambda x:" ".join(token.lemma_ for token in nlp(x) if not token.is_stop and str(token)
                                      not in string.punctuation)).str.lower()

  df['sentence_2'] = df['sentence_2'].apply(lambda x:" ".join(token.lemma_ for token in nlp(x) if not token.is_stop and str(token)
                                      not in string.punctuation)).str.lower()

  df['sentence_1'] = df['sentence_1'].apply(lambda x: ' '.join(text for text in x.split() if len(text)>=2))

  df['sentence_2'] = df['sentence_2'].apply(lambda x: ' '.join(text for text in x.split() if len(text)>=2))

  sentence_1_embeddings = pd.DataFrame(model.encode(df.sentence_1))

  sentence_1_embeddings.columns = ['sent1_'+str(i+1) for i in range(sentence_1_embeddings.shape[1])]

  sentence_2_embeddings = pd.DataFrame(model.encode(df.sentence_2))

  sentence_2_embeddings.columns = ['sent2_'+str(i+1) for i in range(sentence_2_embeddings.shape[1])]

  embedding_df = pd.concat([sentence_1_embeddings,sentence_2_embeddings],axis=1)

  predictions = logreg.predict(embedding_df)

  return predictions.to_dict()
   

if __name__=='__main__':
  app.run('0.0.0.0',debug=True)

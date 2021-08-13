from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def load_model(model_path):
  model = models.load_model(model_path)
  return model

def load_tokenizer(tokenizer_path):
  with open(tokenizer_path , 'rb') as f:
    tokenizer = pickle.load(f)
  return tokenizer

def _comment_to_list(comment):
  comment_list = []
  comment_list.append(comment)
  return comment_list

def _tokenize_pad(comment, tokenizer):
  comment_tokenized = tokenizer.texts_to_sequences(comment)
  cvtd_text = pad_sequences(comment_tokenized, maxlen=100)
  return cvtd_text

def predict_comment(comment, tokenizer, model):
  comment_list = _comment_to_list(comment)
  cvtd_text = _tokenize_pad(comment_list, tokenizer)

  predictions = model.predict(cvtd_text, verbose=0)
  predictions = predictions.tolist()

  prediction_result = {'comment': comment,
                      'toxic': predictions[0][0], 'severe_toxic': predictions[0][1],
                      'obscene': predictions[0][2], 'threat': predictions[0][3],
                      'insult': predictions[0][4], 'identity_hate': predictions[0][5]
                      }

  return prediction_result




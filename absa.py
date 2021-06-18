import pickle

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import model1, model2

# Aspect Based Sentiment Analysis
class ABSA():
  def __init__(self):
    self.device = "cuda" if tf.test.is_gpu_available() else "cpu"

    # load model weights for both Aspect Categorization and Sentiment Analysis
    self.aspect_categorizer = model1
    self.sentiment_classifier = model2

    self.aspect_categorizer.load_weights("models/Aspect_Categorizer_Weights")
    self.sentiment_classifier.load_weights("models/Sentiment_Analyzer_Weights")

    # load tokenizer
    with open('tokenizer/tokenizer.pickle', 'rb') as handle:
        self.tokenizer = pickle.load(handle)

    # load aspects encoder
    with open('encoders/aspects_encoder.pickle', 'rb') as handle:
        self.aspects_encoder = pickle.load(handle)

    # loading aspects encoder
    with open('encoders/sentiments_encoder.pickle', 'rb') as handle:
        self.sentiments_encoder = pickle.load(handle)

  def predict(self, text):
    # preprocess text
    instance = np.asarray([text])
    instance = self.tokenizer.texts_to_sequences(instance)
    instance = pad_sequences(instance, padding='post', maxlen=100)

    # predict aspect and sentiment
    with tf.device(self.device):
      # predict aspect
      aspect_preds = self.aspect_categorizer.predict(instance)
      aspect_pred = aspect_preds.argmax(axis=-1)
      aspect_label = self.aspects_encoder.inverse_transform(aspect_pred)

      # predict sentiment
      sentiment_preds = self.sentiment_classifier.predict(instance)
      sentiment_pred = sentiment_preds.argmax(axis=-1)
      sentiment_label = self.sentiments_encoder.inverse_transform(sentiment_pred)


    return aspect_label.item(), sentiment_label.item().upper()


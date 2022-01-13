from baseline import Baseline
import pandas as pd
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification

class BaselineLyrics(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)
        pd.set_option('display.max_colwidth', None)
        MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
        BATCH_SIZE = 16
        N_EPOCHS = 3

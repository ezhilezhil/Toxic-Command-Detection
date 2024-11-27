from django.db import models
from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models


import numpy as np
import pickle

import json
from PIL import Image


# Testing phase
rf = pickle.load(open(r"C:\Users\dines\Music\FINALCODE\FRONT END\new_project\rf_hatespeech.pkl", 'rb'))



tfidf_feature = pickle.load(open(r"C:\Users\dines\Music\FINALCODE\FRONT END\new_project\tfidf.pkl", 'rb'))


def predict(text,algo): 
	text = [text]
	filter_text = tfidf_feature.transform(text)
	print(filter_text.shape)
	if algo=='rf':
		y_pred=rf.predict(filter_text)
		return y_pred[0]
	else:
		y_pred=rf.predict(filter_text)
		return y_pred[0]


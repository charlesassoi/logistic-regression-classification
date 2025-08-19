#telechargement des bibliotheques necessaires

import pandas as pd
import scikit-learn as sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import logisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
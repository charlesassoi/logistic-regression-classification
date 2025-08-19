#telechargement des bibliotheques necessaires

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#load and prepare the data

data=pd.read_csv("customers-100.csv")

print(data['Index'])

#Text Vectorization

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(data['Website'])
y=data['Index']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#train the logistic regression model
model=LogisticRegression(random_state=42)
model.fit(X_train,y_train)

#model evaluation

y_pred=model.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("accuracy score",accuracy_score(y_test,y_pred))
print("confusion Matrix")
print(f"[{cm[0,0],{cm[0,1]}}]")
print(f"[{cm[1,0],{cm[1,1]}}]")
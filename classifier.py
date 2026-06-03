import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("dataset.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])

y = data["category"]

model = MultinomialNB()
model.fit(X, y)

user_input = input("Enter a message: ")

input_vector = vectorizer.transform([user_input])

prediction = model.predict(input_vector)

print("Category:", prediction[0])
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request

app = Flask(__name__)

class BloomFilter:
    def __init__(self, size=1000):
        self.size = size
        self.bit_array = [0] * size

    def _hashes(self, item):
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16) % self.size
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16) % self.size
        return [h1, h2]

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1

    def check(self, item):
        return all(self.bit_array[h] for h in self._hashes(item))


data = pd.read_csv('email.csv')

X = data.iloc[:, 1:-1]  
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

bf = BloomFilter()
spam_keywords = ["free", "win", "prize", "selected", "congratulations", "offer", "promotion", "lottery", "account", "urgent"]

for word in spam_keywords:
    bf.add(word.lower())

# print("Type your message to test for Spam (q to quit):\n")

@app.route('/',methods=['GET','POST'])
def index():
    result = ''
    if request.method == 'POST':
        msg = request.form['email_msg'].lower().split()

        is_spam_bloom = sum(bf.check(word) for word in msg) >= 2

        features = []
        for col in X.columns:
            count = sum(1 for word in msg if word == col)
            features.append(count)

        input_df = pd.DataFrame([features], columns=X.columns)
        is_spam_model = model.predict(input_df)[0] == 1

        result = "Spam" if is_spam_bloom or is_spam_model else "Not Spam"
        
    return render_template('index.html',result=result)

app.run(debug=True)

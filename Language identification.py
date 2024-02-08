import pandas as pd
import string
import re


data = pd.read_csv("Language identification.csv")
X = data["Text"]
y = data["Language"]

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
y = le.fit_transform(y)

data_list = []
for text in X:
    # removing the symbols 
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    text = text.translate(remove_punct_dict)
    # removing the numbers
    text = re.sub(r"[0-9]+"," ", text)
    # converting the text to lower case
    text = text.lower()
    # appending to data_list
    data_list.append(text)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()



from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X, y)


def LanguageIdentification(text , filename):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The langauge is \""+lang[0]+"\" in the file \""+filename)



def Looping_Files():
   import os

   for filename in os.listdir():
      if filename.endswith(".txt"):
         file = open(filename, encoding="utf8")
         text = file.read()
         LanguageIdentification(text , filename)
         continue
      else:
       continue

Looping_Files()
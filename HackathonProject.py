import pandas as pd
from urllib.request import urlopen
import csv
import re
import spacy
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# nlp = spacy.load("en", disable=["parser", "ner"])
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from spacy.lang import en
# Utilizing google search to get list of websites that are shown from inputting news event
try:
    from googlesearch import search
# Throws exception if module can't be imported
except ImportError:
    print("No module named 'google' found")
query = input("Enter news event")
# Stores all links that are generated from searching the news event
ListofLinks = []
for j in search(query, tld = "co.in", num = 10, stop = 10, pause= 2):
    ListofLinks.append(str(j))
ListofText = []
for i in range(0, len(ListofLinks)):
    response = urlopen(ListofLinks[i])
    html = response.read().decode('utf-8')
    soup = BeautifulSoup(html, features = 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    ListofText.append(text)
header = []
dict_from_csv =  {}
for index in range(0, len(ListofText)):
    header.append('articles' + str(index))
for index in range(0, len(ListofText)):
    with open('articles.csv', 'w') as outfile:
        # print(dict_from_csv)
        mywriter = csv.writer(outfile)
        mywriter.writerow([header[index]])
        mywriter.writerows([ListofText])
        # tokenization and remove punctuations
        modifiedText = word_tokenize(ListofText[index].lower())
        # remove websites and email address
        modifiedText = [re.sub(r"\S+com", "", word) for word in modifiedText]
        modifiedText = [re.sub(r"\S+@\S+", "", word) for word in modifiedText]
        # remove empty spaces
        modifiedText = [word for word in modifiedText if word !=' ']
        # remove the punctuations
        modifiedText = [re.sub(r"\.\?", "", word) for word in modifiedText]
        modifiedText = [re.sub(r"\+\.(1-9)", "", word) for word in modifiedText]
        # keeps track of words that are repeated in each article
        vectorizer = CountVectorizer(stop_words = 'english')
        vectorizer.fit(modifiedText)
        vector = vectorizer.transform(modifiedText)
        df = pd.read_csv('articles.csv', sep = 'delimiter')
        # Trains and tests model
        X_train, X_test, Y_train, Y_test = train_test_split(df[header[index]].values, df[header[index]].values, test_size = 0.3, random_state=1)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.9)
        tfidf_train = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test = tfidf_vectorizer.transform(X_test)
        pac = PassiveAggressiveClassifier(max_iter = 50)
        pac.fit(tfidf_train, Y_train)
        Y_pred = pac.predict(tfidf_test)
        # Extracts score for accuracy of article
        score = 1 - accuracy_score(Y_test, Y_pred)
        print("Accuracy: " + str(score))

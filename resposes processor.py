import nltk
import csv
import os
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def csv_tokenizer(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            response = row['response']
            #print(response)
            tokens = nltk.word_tokenize(response)
            # Filter out tokens i don't want stop words, punctuation, and non-alphabet characters
            filtered_sentence = [w.lower() for w in tokens if not w.lower() in stop_words and w not in string.punctuation and w.isalpha()]
            print(filtered_sentence)


def main():
    csv_tokenizer('C:/Users/jtist/Desktop/gemini_responses.csv')

main()
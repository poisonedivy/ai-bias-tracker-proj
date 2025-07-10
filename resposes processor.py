import nltk
import csv
import os

def csv_tokenizer(filepath):
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            response = row['response']
            #print(response)
            tokens = nltk.word_tokenize(response)
            #for t in tokens:
                #print(t)

def main():
    print("Hello, World!")
    csv_tokenizer('C:/Users/jtist/Desktop/gemini_responses.csv')

main()
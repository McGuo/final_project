import csv
import numpy as np
import pandas as pd
import nltk

# with open('data/UScomments.csv') as csvfile:
#     data = pd.read_csv(csvfile, error_bad_lines=False)
#     data.to_csv("data/processed.csv")

data = pd.read_csv("data/processed.csv")

print(data.head())


sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""

tokens = nltk.word_tokenize(sentence)

print(tokens)

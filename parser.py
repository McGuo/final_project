import csv
import numpy
import pandas

with open('data/UScomments.csv') as csvfile:
    readCSV = pandas.io.parsers.read_csv(csvfile)
    data = []
    for row in readCSV:
        print(row)
        data.append(row)

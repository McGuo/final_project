import csv
import numpy as np
import pandas as pd
import random

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords


def labelViral(row):
    if row['likes'] > 1:
        return ('viral')
    else:
        return ('not viral')

# with open('data/UScomments.csv') as csvfile:
#     data = pd.read_csv(csvfile, error_bad_lines=False, dtype={"comment_id": int, "video_id": str, "comment_text": str, "likes": int, "replies": int})
#     data.to_csv("data/processed.csv")


# The inital list of comments for all videos with type check
data = pd.read_csv("data/processed.csv", error_bad_lines=False,
                   dtype={"comment_id": int, "video_id": str, "comment_text": str, "likes": int, "replies": int})

# Generates a new dataframe which are rows of unique video_ids with max likes and average likes columns
res = data.groupby('video_id').agg({'likes': ['max', 'mean'], 'comment_text': ['count']})
res.columns = ['Max Likes', 'Average Likes', '# of comments']
# print(res)

# Questions to consider: What quantifies as a viral "comment"?
# It should be proportional to the max likes and average likes for a specific video?

# Extracting a single video_id's row of examples
################################################################################
single_video = data.loc[data['video_id'] == "-DGXHMOhXAw"]
single_video['classification'] = single_video.apply(lambda row: labelViral(row), axis=1)
single_video = single_video.drop(['comment_id', 'video_id', 'likes', 'replies'], axis=1)
# print(single_video)

# Constructing the training/testing dataset
################################################################################
msk = np.random.rand(len(single_video)) < 0.8
training_set = single_video[msk]
testing_set = single_video[~msk]

# Structuring Data for Naive Bayes NLTK package
################################################################################
# Stopwords that are considered neutral
stopset = list(set(stopwords.words('english')))


def word_feats(words):
    return dict([(word, True) for word in words.split() if word not in stopset])


posids = []
negids = []

for index, row in training_set.iterrows():
    if row['classification'] == 'viral':
        posids.append(row['comment_text'])
    else:
        negids.append(row['comment_text'])

pos_feats = [(word_feats(f), 'viral') for f in posids]
neg_feats = [(word_feats(f), 'not viral') for f in negids]

trainfeats = pos_feats + neg_feats
classifier = NaiveBayesClassifier.train(trainfeats)

print (classifier.classify(word_feats('I did consider getting someone to animate the limb-mangling parts')))
exit()

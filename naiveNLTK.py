import csv
import numpy as np
import pandas as pd
import random
import nltk.classify
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords


stopset = list(set(stopwords.words('english')))


def labelViral(row, cutoff=1):
    """
    Labels dataframe data with the appropriate classification.
    """
    if row['likes'] >= cutoff:
        return ('viral')
    else:
        return ('not viral')


def word_feats(words):
    """
    Puts a tuple of (word, true) pair into a dictionary for words not in the stopset
    """
    return dict([(word, True) for word in words.split() if word not in stopset])


def formatDataNLTK(dataset):
    """
    Takes in a panda dataframe object and returns a representation usabable by NLTK
    """

    posids = []
    negids = []

    for index, row in dataset.iterrows():
        if row['classification'] == 'viral':
            posids.append(str.lower(row['comment_text']))
        else:
            negids.append(str.lower(row['comment_text']))

    pos_feats = [(word_feats(f), 'viral') for f in posids]
    neg_feats = [(word_feats(f), 'not viral') for f in negids]

    trainfeats = pos_feats + neg_feats
    return trainfeats


# The inital list of comments for all videos with type check
data = pd.read_csv("data/comments_processed.csv", error_bad_lines=False,
                   dtype={"comment_id": int, "video_id": str, "comment_text": str, "likes": int, "replies": int})


# Generates a new dataframe which are rows of unique video_ids with max likes and average likes columns
res = data.groupby('video_id').agg({'likes': ['max', 'mean'], 'comment_text': ['count']})
res.columns = ['Max Likes', 'Average Likes', '# of comments']

#
# for index, row in res.iterrows():
#     if row['classification'] == 'viral':
#         posids.append(str.lower(row['comment_text']))
#     else:
#         negids.append(str.lower(row['comment_text']))

# print(res['# of comments'].max())
# print(res.loc[res['# of comments'].idxmax()])
# exit()


# Questions to consider: What quantifies as a viral "comment"?
# It should be proportional to the max likes and average likes for a specific video?

# Extracting a single video_id's row of examples
################################################################################
single_video = data.loc[data['video_id'] == "4X6a3G_0HjY"]
single_video['classification'] = single_video.apply(lambda row: labelViral(row), axis=1)
single_video = single_video.drop(['comment_id', 'video_id', 'likes', 'replies'], axis=1)
# print(single_video)

# Constructing the training/testing dataset
################################################################################
msk = np.random.rand(len(single_video)) < 0.8
training_set = formatDataNLTK(single_video[msk])

test_set = formatDataNLTK(single_video[~msk])
test_set_dataframe = single_video[~msk]

# Using the Naive Bayes NLTK package
################################################################################
classifier = NaiveBayesClassifier.train(training_set)
classifier.show_most_informative_features()
print("Our model has an accuracy of:", nltk.classify.accuracy(classifier, test_set))

# print (classifier.classify_many(test_set))

# print(classifier.classify(word_feats('hello world')))
#
# for index, row in test_set_dataframe.iterrows():
#     print(classifier.classify(word_feats(row['comment_text'])))

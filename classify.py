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
        try:
            if row['classification'] == 'viral':
                posids.append(str.lower(row['comment_text']))
            else:
                negids.append(str.lower(row['comment_text']))
        except:
            pass

    pos_feats = [(word_feats(f), 'viral') for f in posids]
    neg_feats = [(word_feats(f), 'not viral') for f in negids]

    trainfeats = pos_feats + neg_feats
    return trainfeats


def executeOrder66():

    # The inital list of comments for all videos with type check
    data = pd.read_csv("data/comments_processed.csv", error_bad_lines=False,
                       dtype={"comment_id": int, "video_id": str, "comment_text": str, "likes": int, "replies": int})

    unique_videos = data.groupby('video_id').agg({'likes': ['max', 'mean'], 'comment_text': ['count']})
    unique_videos.columns = ['Max Likes', 'Average Likes', '# of comments']

    # testModel(data, unique_videos, "guessNotViral")
    testModel(data, unique_videos, "naiveBayes")


def testModel(data, unique_videos, model):
    """
    Tests the data under a specific model.
    """
    total_count = 0
    accuracy = 0
    correct = 0
    total_classification = 0
    for index, row in unique_videos.iterrows():
        try:
            single_video = data.loc[data['video_id'] == index]
            single_video['classification'] = single_video.apply(lambda example: labelViral(example, row['Average Likes']), axis=1)
            msk = np.random.rand(len(single_video)) < 0.8
            test_set = single_video[~msk]
            training_set = single_video[msk]

            if model is "guessNotViral":
                for index, row in test_set.iterrows():
                    if (row['classification'] == 'not viral'):
                        correct += 1
                        total_classification += 1
                    else:
                        total_classification += 1
                accuracy += correct / total_classification
                total_count += 1

            elif model is "naiveBayes":
                classifier = NaiveBayesClassifier.train(formatDataNLTK(training_set))
                accuracy += nltk.classify.accuracy(classifier, formatDataNLTK(test_set))
                total_count += 1

            print("current average accuracy of:", accuracy / total_count, "with", total_count, "videos analyzed")

        except:
            pass

    print('After analyzing', total_count, "different videos and their comments, we have an average accuracy of", accuracy / total_count, "using", model)


executeOrder66()

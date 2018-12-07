import csv
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import ComplementNB


def labelViral(row, cutoff=1):
    """
    Labels a row with the appropriate classification.
    """
    if row['likes'] >= cutoff:
        return ('viral')
    else:
        return ('not viral')


def runModel():
    dataframe = pd.read_csv("data/comments_processed.csv", error_bad_lines=False,
                            dtype={"comment_id": int, "video_id": str, "comment_text": str, "likes": int, "replies": int})

    unique_videos = dataframe.groupby('video_id').agg({'likes': ['max', 'mean'], 'comment_text': ['count']})
    unique_videos.columns = ['Max Likes', 'Average Likes', '# of comments']

    total_count = 0

    total_accuracy = 0
    for index, row in unique_videos.iterrows():
        try:
            data = []
            data_labels = []
            single_video = dataframe.loc[dataframe['video_id'] == index]

            for index, row2 in single_video.iterrows():
                row2['classification'] = labelViral(row2, row['Average Likes'])
                data.append(row2['comment_text'])
                data_labels.append(row2['classification'])

            vectorizer = CountVectorizer(analyzer='word', lowercase=False,)
            features = vectorizer.fit_transform(data)
            features_nd = features.toarray()

            X_train, X_test, y_train, y_test = train_test_split(features_nd, data_labels, random_state=1234)

            # log_model = LogisticRegression(solver='lbfgs')
            # model = "Logistic Regression"

            log_model = ComplementNB()
            model = "Complement Naive Bayes"

            log_model = log_model.fit(X=X_train, y=y_train)

            y_pred = log_model.predict(X_test)

            # j = random.randint(0, len(X_test) - 7)
            # for i in range(j, j + 7):
            #     print(y_pred[0])
            #     ind = features_nd.tolist().index(X_test[i].tolist())
            #     print(data[ind].strip())

            total_count += 1
            total_accuracy += accuracy_score(y_test, y_pred)

        except:
            pass
        print ("current average accuracy of", total_accuracy / total_count, "with", total_count, "videos ran")

    print()
    print()
    print("Using", model, "we had an average accuracy score of ",
          total_accuracy / total_count, "over", total_count, "unique videos and its comments")


runModel()

import csv
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn import linear_model

from sklearn.linear_model import LogisticRegressionCV


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

    total_correct = 0
    total = 0

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

            X_train, X_test, y_train, y_test = train_test_split(features_nd, data_labels)

            # Model Selection
            ################################################################################

            # log_model = LogisticRegression(solver='lbfgs')
            # model = "Logistic Regression"

            log_model = LogisticRegressionCV(cv=10, multi_class='multinomial')
            model = "Logistic Regression with Cross Validation and no class_weight"

            # log_model = ComplementNB()
            # model = "Complement Naive Bayes"

            # log_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            # model = "Multi-layer Perceptron"

            # log_model = svm.SVC(gamma='scale')
            # model = "Support Vector Machines"

            # log_model = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
            # model = "Linear classifiers with SGD training"

            ################################################################################

            log_model = log_model.fit(X=X_train, y=y_train)

            y_pred = log_model.predict(X_test)

            # j = random.randint(0, len(X_test) - 7)
            # for i in range(j, j + 7):
            #     print(y_pred[0])
            #     ind = features_nd.tolist().index(X_test[i].tolist())
            #     print(data[ind].strip())

            # correct / total = accuracy

            total_correct += len(y_test) * accuracy_score(y_test, y_pred)
            total += len(y_test)
            total_count += 1

        except:
            pass
        print ("current average accuracy of", total_correct / total, "with", total_count, "videos ran")

    print()
    print()
    print("Using", model, "we had an average accuracy score of",
          total_correct / total, "with", total_correct, "comments being labeled correctly out of", total)

    print()
    print()


runModel()

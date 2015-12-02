from abc import abstractmethod

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from preprocess import Preprocess
import random
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self):
        return

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def train(self, data_train):
        pass

    def featurize(self, data):
        return data

    @staticmethod
    def calc_error(Y_predicted, Y_actual):
        if len(Y_predicted) != len(Y_actual):
            raise Exception("Y_predicted length does not match Y_actual length")

        # calculate error
        count_wrong = 0.0
        for i in range(0, len(Y_predicted)):
            if Y_predicted[i] != Y_actual[i]:
                count_wrong += 1.0
        return count_wrong/len(Y_predicted)


class BaselineModel(Model):
    def get_name(self):
        return 'Baseline (always guess italian)'

    # Simply always predict italian, the most popular cuisine in our training data set
    def predict(self, X):
        if isinstance(X, list):
            return ["italian" for i in range(0, len(X))]
        else:
            return "italian"

class RandomGuessModel(Model):
    CUISINES = ["italian"]

    def __init__(self, cuisines):
        self.CUISINES = cuisines
        random.seed(69)
        return

    def get_name(self):
        return 'Random Guess'

    # Simply always predict italian, the most popular cuisine in our training data set
    def predict(self, X):

        if isinstance(X, list):
            return [random.choice(self.CUISINES) for i in range(0, len(X))]
        else:
            return random.choice(self.CUISINES)

class CustomScoreModel(Model):
    memo = {}

    def get_name(self):
        return "Custom Score Model"

    def calculate_score1(self, cuisines_list, cuisine_set, cuisine, ingredient_list, ingredient, counts):
        score = 0.0
        recipes_in_cuisine = 0.0
        recipes_not_in_cuisine = 0.0

        for curr_cuisine in cuisine_set:
            if curr_cuisine == cuisine:
                recipes_in_cuisine += len(cuisines_list[curr_cuisine])
            else:
                recipes_not_in_cuisine += len(cuisines_list[curr_cuisine])


        if (cuisine, ingredient) in self.memo.keys():
            score += self.memo[(cuisine, ingredient)]
            return score

        times_ingredient_in_cuisine = counts[(cuisine, ingredient)]
        times_ingredient_not_in_cuisine = len(ingredient_list[ingredient]) - times_ingredient_in_cuisine

        self.memo[(cuisine, ingredient)] = times_ingredient_in_cuisine/recipes_in_cuisine - times_ingredient_not_in_cuisine/recipes_not_in_cuisine
        score += times_ingredient_in_cuisine/recipes_in_cuisine - times_ingredient_not_in_cuisine/recipes_not_in_cuisine
        return score


    def calculate_score(self, cuisines_list, cuisine_set, cuisine, ingredient_list, datum, counts):
        score = 0.0
        recipes_in_cuisine = 0.0
        recipes_not_in_cuisine = 0.0

        for curr_cuisine in cuisine_set:
            if curr_cuisine == cuisine:
                recipes_in_cuisine += len(cuisines_list[curr_cuisine])
            else:
                recipes_not_in_cuisine += len(cuisines_list[curr_cuisine])

        for ingredient in datum['ingredients']:
            if (cuisine, ingredient) in self.memo.keys():
                score += self.memo[(curr_cuisine, ingredient)]
                continue

            times_ingredient_in_cuisine = counts[(cuisine, ingredient)]
            times_ingredient_not_in_cuisine = len(ingredient_list[ingredient]) - times_ingredient_in_cuisine

            self.memo[(cuisine, ingredient)] = times_ingredient_in_cuisine/recipes_in_cuisine - times_ingredient_not_in_cuisine/recipes_not_in_cuisine
            score += times_ingredient_in_cuisine/recipes_in_cuisine - times_ingredient_not_in_cuisine/recipes_not_in_cuisine
        return score


    #TODO: BROKEN. ADAPT TO NEW DESIGN.
    def predict(self, p, data):
        predictions = []
        print "Length of dataset predictions", len(data)
        for datum in data:
            curr_best_prediction = "ERROR"
            curr_best_score = float("-inf")
            for cuisine in p.cuisines_set:
                score = self.calculate_score(p.cuisines, p.cuisines_set, cuisine, p.ingredient_list, datum, p.counts)
                if score > curr_best_score:
                    curr_best_score = score
                    curr_best_prediction = cuisine
            predictions.append(curr_best_prediction)
        return predictions



class RandomForestModel(Model):
    random_forest = RandomForestClassifier()
    label_encoder = LabelEncoder()
    TOKEN_INGREDIENT_SEPARATOR = "|"
    ngram_vectorizer = None

    def __init__(self):
        # initialize ngram vectorizer with custom tokenizer
        self.ngram_vectorizer = CountVectorizer(tokenizer=lambda s: s.split(self.TOKEN_INGREDIENT_SEPARATOR))
        return

    def get_name(self):
        return "Random Forest Classifier with bag-of-ingredients (unigrams)"

    def featurize(self, data):
        data_X = [self.TOKEN_INGREDIENT_SEPARATOR.join(r['ingredients']) for r in data]
        return self.ngram_vectorizer.transform(data_X).toarray()

    def train(self, data_train):
        X_train = [self.TOKEN_INGREDIENT_SEPARATOR.join(recipe['ingredients']) for recipe in data_train]
        Y_train = [recipe['cuisine'] for recipe in data_train]


        # transform features
        X_train = self.ngram_vectorizer.fit_transform(X_train).toarray()
        # transform labels to distinct ints
        Y_train = self.label_encoder.fit_transform(Y_train)

        self.random_forest.fit(X_train, Y_train)
        return

    def predict(self, X):
        Y_predicted = self.random_forest.predict(X)
        # transform labels back to text
        return self.label_encoder.inverse_transform(Y_predicted)

class LogisticRegressionModel(Model):
    logistic_regression = LogisticRegression()
    label_encoder = LabelEncoder()
    TOKEN_INGREDIENT_SEPARATOR = "|"
    ngram_vectorizer = None

    def __init__(self):
        # initialize ngram vectorizer with custom tokenizer
        self.ngram_vectorizer = CountVectorizer(tokenizer=lambda s: s.split(self.TOKEN_INGREDIENT_SEPARATOR))
        return

    def get_name(self):
        return "Logistic Regression with bag-of-ingredients (unigrams)"

    def featurize(self, data):
        data_X = [self.TOKEN_INGREDIENT_SEPARATOR.join(r['ingredients']) for r in data]
        return self.ngram_vectorizer.transform(data_X).toarray()

    def train(self, data_train):
        X_train = [self.TOKEN_INGREDIENT_SEPARATOR.join(recipe['ingredients']) for recipe in data_train]
        Y_train = [recipe['cuisine'] for recipe in data_train]


        # transform features
        X_train = self.ngram_vectorizer.fit_transform(X_train).toarray()
        # transform labels to distinct ints
        Y_train = self.label_encoder.fit_transform(Y_train)

        self.logistic_regression.fit(X_train, Y_train)
        return

    def predict(self, X):
        Y_predicted = self.logistic_regression.predict(X)
        # transform labels back to text
        return self.label_encoder.inverse_transform(Y_predicted)

class LogisticRegressionModelTfidf(LogisticRegressionModel):
    tfidf = None

    def __init__(self, sublinear_tf=False, norm=None):
        super(self.__class__, self).__init__()
        self.tfidf = TfidfTransformer(sublinear_tf=sublinear_tf, norm=norm)


    def get_name(self):
        return "Logistic Regression with bag of ingredients (unigrams) and tf-idf"

    def featurize(self, data):
        data_X = [self.TOKEN_INGREDIENT_SEPARATOR.join(r['ingredients']) for r in data]
        counts = self.ngram_vectorizer.transform(data_X)
        return self.tfidf.transform(counts).toarray()

    def train(self, data_train):
        X_train = [self.TOKEN_INGREDIENT_SEPARATOR.join(recipe['ingredients']) for recipe in data_train]
        Y_train = [recipe['cuisine'] for recipe in data_train]


        # transform features
        X_train = self.ngram_vectorizer.fit_transform(X_train)
        X_train = self.tfidf.fit_transform(X_train).toarray()
        # transform labels to distinct ints
        Y_train = self.label_encoder.fit_transform(Y_train)

        self.logistic_regression.fit(X_train, Y_train)
        return

MILLISECS_TO_SECS_DIVISOR = 1000

def get_confusion_matrix(model, data, cuisines_set):
    labels = list(cuisines_set)
    C = [[0.0 for i in range(len(labels))] for j in range(len(labels))]
    label_counts = [0.0] * len(labels)
    for datum in data:
        j = labels.index(datum['cuisine'])
        i = labels.index(model.predict(model.featurize([datum]))[0])
        C[i][j] += 1.0
        label_counts[j] += 1.0

    for i in range(0, len(labels)):
        for j in range(0, len(labels)):
            C[i][j] = C[i][j]/label_counts[j]
    return (C, labels)

def print_confusion_matrix_latex(cuisine_mapping, C):
    print "<htm><body><table>"
    header = "<tr>" + "<td>Confusion Matrix</td>"
    for i in range(0, len(cuisine_mapping)):
        header += "<td>" + cuisine_mapping[i] + "</td>"
    header += "</tr>"
    print header
    
    for i in range (0, len(cuisine_mapping)):
        row = "<tr>" + "<td>" + cuisine_mapping[i] + "</td>"
        for j in range(0, len(cuisine_mapping)):
            row += "<td>"
            if j == len(cuisine_mapping) - 1:
                row += str("{0:.3f}".format(C[i][j])) 
            else:
                row += str("{0:.3f}".format(C[i][j]))
            row += "</td>"
        row += "</tr>"
        print row
    print "</table></body></html>" 


def main():
    print "====================="
    print "What Cuisine?"
    print "Predicting cuisines of recipes using multi-class classification"
    print "by Edward Wong, Maya Nyayapati, Hannes Holste"
    print "=====================\n\n"

    # preprocess
    print "Preprocessing data (reading, cleansing)..."
    start = time.time()
    p = Preprocess('data/train.json')
    data = p.data
    total = len(data)
    TRAIN_SET_SIZE = int(total * 0.8)
    end = time.time()
    print "\tDone (%f s)\n" % ((end - start))

    print "Splitting data into training set and validation set."
    # split data into training and validation set

    data_train = data[0:TRAIN_SET_SIZE]
    data_validation = data[TRAIN_SET_SIZE:total]

    # Run models
    models = [#BaselineModel(),
              #RandomGuessModel(cuisines=list(p.cuisines_set)),
              #RandomForestModel(),
              #LogisticRegressionModel(),
              LogisticRegressionModelTfidf(sublinear_tf=True, norm="l2")
              ]

    # calculate correct labels for error calc later
    Y_actual_validation = [datum['cuisine'] for datum in data_validation]
    Y_actual_train = [datum['cuisine'] for datum in data_train]

    print "Models to be run: ", ", ".join([model.get_name() for model in models])
    for model in models:
        print "====================="
        print "Model: ", model.get_name()

        # train
        start = time.time()
        print "\tTraining on %d training set examples" % len(data_train)
        model.train(data_train)
        end = time.time()
        print "\tDone (%d s)\n" % ((end - start))

        # make predictions and calculate error

        # training set
        print "\tMaking %d predictions on training set..." % len(data_train)
        start = time.time()
        Y_predicted = model.predict(model.featurize(data_train))
        end = time.time()
        print "\tDone (%d s)\n" % ((end - start))

        print "\t Calculating training set error..."
        print "\t Error rate: %f" % model.calc_error(Y_predicted, Y_actual_train)

        # validation set
        print "\tMaking %d predictions on validation set..." % len(data_validation)
        start = time.time()
        Y_predicted = model.predict(model.featurize(data_validation))
        end = time.time()
        print "\tDone (%d s)\n" % ((end - start))

        print "\t Calculating validation set error..."
        print "\t Error rate: %f" % model.calc_error(Y_predicted, Y_actual_validation)

        # how to use this
        (C, mapping) = get_confusion_matrix(model, data_validation, p.cuisines_set)
        print_confusion_matrix_latex(mapping, C)


main()


import random
import time
from abc import abstractmethod
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import preprocess
from preprocess import Preprocess


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

def predict_and_output_csv(model):
    print "Training model %s, and predicting, and outputting predictions" % model.get_name()
    print "Model: ", model.get_name()

    raw_test_data = Preprocess.parseData("data/test.json")
    data_test = list()
    ids = []
    for recipe in raw_test_data:
        clean_recipe = defaultdict(list)
        ids.append(recipe["id"])
        for ingredient in recipe["ingredients"]:
            clean_ingredient = Preprocess.process_ingredient(ingredient)
            clean_recipe["ingredients"].append(clean_ingredient)
        data_test.append(clean_recipe)

    print "\tMaking %d predictions on test set..." % len(data_test)
    Y_predicted = model.predict(model.featurize(data_test))

    # format data for output
    print "\tOutputting predictions..."
    header = "id,cuisine"
    with open("test_set_predictions.csv","w") as f:
        f.write(header + "\n")
        i = 0
        for prediction in Y_predicted:
            id = ids[i]
            predicted_cuisine = prediction
            f.write(str(id) + "," + predicted_cuisine + "\n")
            i += 1
    return


def main():
    print "====================="
    print "What Cuisine?"
    print "Predicting cuisines of recipes using multi-class classification"
    print "by Edward Wong, Maya Nyayapati, Hannes Holste"
    print "=====================\n\n"

    # preprocess
    print "Preprocessing data (reading, cleansing)..."
    start = time.time()
    p = Preprocess("data/train.json")
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
    models = [BaselineModel(),
              RandomGuessModel(cuisines=list(p.cuisines_set)),
              RandomForestModel(),
              LogisticRegressionModel(),
              LogisticRegressionModelTfidf(sublinear_tf=True, norm="l2")
              ]

    # calculate correct labels for error calc later
    Y_actual_validation = [datum['cuisine'] for datum in data_validation]
    Y_actual_train = [datum['cuisine'] for datum in data_train]

    best_model = None
    lowest_error = float("inf")

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

        error = model.calc_error(Y_predicted, Y_actual_validation)

        print "\t Calculating validation set error..."
        print "\t Error rate: %f" % error

        # save best model
        if best_model == None or error < lowest_error:
            lowest_error = error
            best_model = model

    print "\n\nBest model: %s with error of %f" % (best_model.get_name(), lowest_error)

    print "\nTraining best model and running on test set, then outputting...\n"

    predict_and_output_csv(model=best_model)

main()


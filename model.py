from preprocess import Preprocess
import random
import math

memo = {}


def calculate_score1(cuisines_list, cuisine_set, cuisine, ingredient_list, ingredient, counts):
    score = 0.0
    recipes_in_cuisine = 0.0
    recipes_not_in_cuisine = 0.0

    for curr_cuisine in cuisine_set:
        if curr_cuisine == cuisine:
            recipes_in_cuisine += len(cuisines_list[curr_cuisine])
        else:
            recipes_not_in_cuisine += len(cuisines_list[curr_cuisine])

   
    if (cuisine, ingredient) in memo.keys():
        score += memo[(cuisine, ingredient)]
        return score

    times_ingredient_in_cuisine = counts[(cuisine, ingredient)]
    times_ingredient_not_in_cuisine = len(ingredient_list[ingredient]) - times_ingredient_in_cuisine

    memo[(cuisine, ingredient)] = times_ingredient_in_cuisine/recipes_in_cuisine - times_ingredient_not_in_cuisine/recipes_not_in_cuisine
    score += times_ingredient_in_cuisine/recipes_in_cuisine - times_ingredient_not_in_cuisine/recipes_not_in_cuisine
    return score


def calculate_score(cuisines_list, cuisine_set, cuisine, ingredient_list, datum, counts):
    score = 0.0
    recipes_in_cuisine = 0.0
    recipes_not_in_cuisine = 0.0

    for curr_cuisine in cuisine_set:
        if curr_cuisine == cuisine:
            recipes_in_cuisine += len(cuisines_list[curr_cuisine])
        else:
            recipes_not_in_cuisine += len(cuisines_list[curr_cuisine])

    for ingredient in datum['ingredients']:
        ingredient = ingredient.lower()
        if (cuisine, ingredient) in memo.keys():
            score += memo[(curr_cuisine, ingredient)]
            continue
    
        times_ingredient_in_cuisine = counts[(cuisine, ingredient)]
        times_ingredient_not_in_cuisine = len(ingredient_list[ingredient]) - times_ingredient_in_cuisine

        memo[(cuisine, ingredient)] = times_ingredient_in_cuisine/recipes_in_cuisine - times_ingredient_not_in_cuisine/recipes_not_in_cuisine
        score += times_ingredient_in_cuisine/recipes_in_cuisine - times_ingredient_not_in_cuisine/recipes_not_in_cuisine
    return score

def tfidf(cuisines_list, cuisine_set, cuisine, ingredient_list, datum, counts):
    score = 0.0
    length = 0.0
    for c in cuisine_set:
        length += len(cuisines_list[c])

    for ingredient in datum['ingredients']:
        ingredient = ingredient.lower()
        tf = counts[(cuisine, ingredient)]
        df = float(len(ingredient_list[ingredient]))
        idf = math.log(length/df)
        score += tf * idf
    return score




def get_prediction(p, data):
    predictions = []
    print "Length of dataset predictions", len(data)
    for datum in data:
        curr_best_prediction = "OOPS"
        curr_best_score = float("-inf")
        for cuisine in p.cuisines_set:
            score = tfidf(p.cuisines, p.cuisines_set, cuisine, p.ingredient_list, datum, p.counts)
            if score > curr_best_score:
                curr_best_score = score
                curr_best_prediction = cuisine
        predictions.append(curr_best_prediction)
    return predictions

def calc_error(predictions, label):
    count_wrong = 0.0
    for i in range(0, len(predictions)):
        if not(predictions[i] == label[i]):
            count_wrong += 1.0
    return count_wrong/len(predictions)

def main():
    p = Preprocess()
    data = p.data
    validation_set = data[:5000]
    predictions = get_prediction(p, validation_set)
    correct_answers = [datum['cuisine'] for datum in validation_set]
    print "Error Rate:", calc_error(predictions, correct_answers)
    baseline = ['italian'] * len(validation_set)
    print "Baseline Error Rate:", calc_error(baseline, correct_answers)
    p.print_info()


main()


import json
from pprint import pprint
from collections import defaultdict

import regex

import sys

reload(sys)
sys.setdefaultencoding("utf-8")


class Preprocess:
    cuisines = defaultdict(list)
    cuisines_set = set()
    ingredient_list = defaultdict(list)
    ingredient_set = set()
    data = []
    counts = defaultdict(int)
    food_adjectives = []

    def __init__(self):
        self.load_food_adjectives()
        self.load_data(func_process_ingredient=self.process_ingredient)


    def load_food_adjectives(self):
        for l in open('data/food_adjectives.txt', 'r'):
            if len(l) > 0:
                self.food_adjectives.append(l.replace("\n", ""))

    def parseData(self, fname):
        with open(fname) as data_file:
            data = json.load(data_file)
            return data

    def process_ingredient(self, ingredient_name):
        # strip unicode e.g. \u2012
        ingredient_name = ingredient_name.decode('unicode_escape').encode('ascii','ignore')

        # strip punctuation
        ingredient_name = regex.sub(ur"\p{P}+", "", ingredient_name)

        # strip parantheses and their contents
        # e.g. ( 3  oz.) tomato paste
        ingredient_name = regex.sub(ur"\(.+\)", "", ingredient_name)


        # remove food adjectives
        for adjective in self.food_adjectives:
            # only substitute adjectives if they appear as independent words, e.g. replace "or" in "toast or bacon", not "pORk"
            ingredient_name = regex.sub(u"(" + adjective + " |" + adjective + "or)", " ", ingredient_name)

        # remove excess spaces
        ingredient_name = " ".join(ingredient_name.split())

        # remove preceding and trailing spaces
        return ingredient_name.strip()

    def load_data(self, func_process_ingredient=lambda s: s):
        default_data_dir = "data/train.json"
        self.data = self.parseData(default_data_dir)
        clean_data = list()

        # For each recipe...
        for datum in self.data:
            clean_datum = dict()

            # process cuisine
            cuisine = datum['cuisine'].lower()
            clean_datum["cuisine"] = cuisine

            self.cuisines[cuisine].append(datum)
            self.cuisines_set.add(cuisine)

            # process ingredients
            ingredients = datum['ingredients']
            clean_ingredients = list()

            for ingredient in ingredients:
                # always lowercase ingredient names
                ingredient = ingredient.lower()

                # clean up ingredient string
                ingredient = func_process_ingredient(ingredient)

                clean_ingredients.append(ingredient)

                self.counts[(cuisine,ingredient)] += 1
                self.ingredient_list[ingredient].append(datum)
                self.ingredient_set.add(ingredient)

            clean_datum["ingredients"] = clean_ingredients

            clean_data.append(clean_datum)

        self.data = clean_data


    def print_info(self):
        print "Number of recipes:", len(self.data)

        cuisine_count = []
        for cuisine in self.cuisines_set:
            cuisine_count.append((cuisine, len(self.cuisines[cuisine])))
        cuisine_count.sort(key=lambda tup: tup[1])
        for (cuisine, count) in cuisine_count:
            print cuisine.title(), count

        ingredient_count = []
        for ingredient in self.ingredient_set:
            ingredient_count.append((ingredient, len(self.ingredient_list[ingredient])))
        print len(ingredient_count), "different kinds of ingredients"

        ingredient_count.sort(key=lambda tup: tup[1])
        for i in range(0, 25):
            print ingredient_count[i][0], ingredient_count[i][1]

        ingredient_count.reverse()
        for i in range(0, 25):
            print ingredient_count[i][0], ingredient_count[i][1]

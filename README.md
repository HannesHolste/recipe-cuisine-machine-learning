# What Cuisine? Classifying cuisines based on recipes using multi-class classification

# Overview
We consider the problem of predicting the cuisine of
a recipe given a list of ingredients. Such classification can help
food databases and recipe recommender systems autonomously
categorize new recipes based on their ingredients. Results of
our evaluations show that a test set classification accuracy of
at least 77.87% is possible given a training set of 39,774 recipes,
significantly surpassing the accuracy of a baseline predictor that,
for each recipe, trivially guesses the most common cuisine.

Keywords: food, recipe, data mining, machine learning, classification

# Introduction
Recipe search and recommendation websites such as
Yummly are growing in popularity [1]. When users contribute
new recipes to these websites, they are faced with a large
number of fields that require manual input of ingredient
lists, cooking steps, cuisine types, and descriptions, among
other data. The presence of a large number of input fields is
problematic, as an increase in form inputs has been shown to
heighten the probability that a user, out of frustration, abandons
a form entirely [2].

In this context, we present a machine learning strategy to
automatically categorize recipes by cuisine. Automatic classification has three benefits.

First, machine-driven classification reduces required user
input, thus potentially decreasing form abandonment rates.

Second, it is also useful in developing a notion of cuisine
similarity, allowing restaurant recommendation systems such
as Yelp to compare user cuisine preferences with restaurant
meal offerings, thus potentially leading to more relevant suggestions.

Finally, automatic cuisine labeling can also help users discover
what cuisines their custom, un-categorized recipes are
likely to belong to, allowing them to label their recipes with
less cognitive effort.

# Link to Published Paper
This repository contains the code used to build and test the machine learning models used in the published paper: [Link to PDF publication](http://www.hannesholste.com/publications/CSE190_ML_Recipe_Cuisines_Paper-2015.pdf)

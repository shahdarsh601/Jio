# Sentiment Analysis API

This project implements a sentiment analysis tool that categorizes product or movie reviews as Positive, Negative, or Neutral. It uses the IMDb dataset for training and a Logistic Regression model for classification.

## Files

- **train_model.py**: Trains the sentiment analysis model and saves it.
- **api.py**: Flask application that provides an API endpoint for sentiment analysis and contains functions to preprocess the text data.
- **README.md**: This file contains information about the project

## Setup

- import nltk
- import pandas
- Download the dataset from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project directory.

## Usage

- Run a Flask application such as Postman
- Make a POST request to the `/detect` endpoint with a JSON payload containing the sentence (could be of your choice):
  
    {
  
         "review": "The movie was amazing!"
  
    }

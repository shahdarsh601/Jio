# SingularPlural Detection API

This project contains a Flask API that detects singular and plural nouns in a given sentence. It uses NLTK for natural language processing.

## Files

- **singular_plural_detector.py**: Contains the `SingularPluralDetector` class with methods to detect singular and plural nouns.
- **api.py**: Flask API containing an endpoint to detect singular and plural nouns using the `SingularPluralDetector` class.
- **README.md**: This file contains information about the project.

## Setup

- import ntlk
- import flask

## Usage

- Run a Flask application such as Postman
- Make a POST request to the `/detect` endpoint with a JSON payload containing the sentence (could be of your choice):
  
    {
        "sentence": "The cats are running in the gardens."
    }

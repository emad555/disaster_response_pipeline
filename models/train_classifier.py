# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification

import nltk

nltk.download(['wordnet', 'punkt', 'stopwords'])



def load_data(database_filepath):
    '''
    The function is used to load the data from the sql files
    
    parmeters: 
    database_filepath (Str): the path of the sql file of the dataset
   
    Returns three parameters: X: the masseges values ,Y: classification label, category_names as a category names, 
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse",engine)
    df.describe()
    df = df.drop(['child alone'],axis=1)
    X = df['message']  # Message Column
    Y = df.iloc[:,5:] 
    Y=Y.astype('int')
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    The function is used natural language processing techniques to make the training model easier
    
    parmeters: 
    text (Str): the message input
   
    Returns the messages as token 
    '''
    #normalize the text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    
    #remove stops
    stop = stopwords.words('english')
    words = [t for t in words if t not in stop]
    
    # Lemmatize
    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in words]
    
    return tokens


def build_model():
    '''
    The function is to build the machine learning model for the classification of the messages
    We used GridSearchCV to find the best paramaeters to improve the accurcy

    Returns the model
    '''
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf': (True, False),
        'vect__max_df': (0.5,0.75),
        'clf__estimator__n_estimators': [10]
        }
    cv = GridSearchCV(model, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    The function is used to evaluate the generated model
    
    parmeters: 
    model (float): the final weights of the model
    X_test (float), y_test (float): the test data
    category_names (Str)

    '''

        #Testing the model
    # Printing the classification report for each label

    y_pred = model.predict(X_test)
    i = 0
    for col in y_test:
        print('Feature {}: {}'.format(i+1, col))
        print(classification_report(y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    

def save_model(model, model_filepath):
    '''
    The function is to save the model in a spacific folder to use it
    parmeters:
    model (float): the final weights of the model
    model_filepath (Str): the folder where to save the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
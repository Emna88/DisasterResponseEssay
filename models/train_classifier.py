import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import statements
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

import pickle

def load_data(database_filepath):
    
    """
    Funtion to read df from sqlite3
    INPUT
    string - path ot the database filename
    OUTPUT
    X - Feature for training  
    Y - Feature to be evaluated
    category_names - Array of category names.
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponseTable',engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = Y.columns.values
    return X,Y,category_names

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    
    """
    Funtion to tokenize text
    INPUT
    string - input text
    OUTPUT
    clean_tokens - cleaned tokenized text.
    """
    
    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Tokenize words
    tokens = word_tokenize(text)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    
    # Normalize text
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    # Custom transformer to extract starting verb
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

class StartingNounExtractor(BaseEstimator, TransformerMixin):
    # Custom transformer to extract starting Noun
    def starting_noun(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['NN', 'NNS', 'NNP', 'NNPS'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_noun)
        return pd.DataFrame(X_tagged)

def build_model():
    
    """
    Funtion to build model pipeline 
    INPUT
    None
    OUTPUT
    cv - model instance 
    """
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
              'classifier__estimator__n_estimators': [5]}

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2,scoring='f1_micro', n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """
    Funtion to evaluate model performance
    INPUT
    
    model - model to train on the dataset
    X_test - test data for training
    Y_test - test data for evaluating
    category_names - Array of category names
    OUTPUT
    
    Print accuracy score and classification report
    
    """
    # Predict on data
    Y_pred = model.predict(X_test)
    
    accuracy = (Y_pred == Y_test).mean()
    report = classification_report(Y_test.values, Y_pred, target_names=category_names)
    
    print (accuracy)    
    print ('\n', report)


def save_model(model, model_filepath):

    """
    Funtion to save model as pkl file
    INPUT
    
    model - model to save
    model_filepath - Path to save the trained model
    OUTPUT
    """    
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
                
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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

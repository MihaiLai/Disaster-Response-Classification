# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pickle

def load_data(database_filepath):
    '''load data from database table and return features:X,labels:y,labels'name columns
    keywords:
        database_filepath:path of database
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM InsertTableName", engine)
    
    X = df['message'].values
    y_df = df.drop(['id','message','original','genre'],axis=1)
    columns = y_df.columns
    y = y_df.values
    return X,y,columns


def tokenize(text):
    '''process text data
    keywords:
        text:input text
    '''
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # word tokenize
    tokens = word_tokenize(text)
    # lemmatiz
    lem = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_tok = lem.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''build meachine learning model and return it
    '''
    pipeline = Pipeline([
        ('vec',CountVectorizer(stop_words='english',tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state=0))))
    ])

    parameters = {
        'vec__ngram_range':[(1,1),(1,2)],
        'vec__max_df':[0.6,0.8,1],
        'tfidf__use_idf':[False,True],
        'clf__estimator__estimator__C':[0.6,0.8,1]
    }
    # use gridsearch to find best params of model
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv




def evaluate_model(model, X_test, Y_test, category_names):
    '''Report the f1 score, precision and recall for each output category of the dataset
    keywords:
    model:meachine learning model
    X_test:test features
    Y_test: test labels
    category_names:label's name
    '''
    y_pred = model.predict(X_test)
    for index,column in enumerate(category_names):
        print('label: ' + column)
        print(classification_report(Y_test[:,index], y_pred[:,index]))
    print(model.score(X_test,Y_test))


def save_model(model, model_filepath):
    '''save model after training it
    keywords:
    model_filepath:save model to this path
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
        print(model.best_params_)
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
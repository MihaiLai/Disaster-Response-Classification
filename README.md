# Disaster Response Classification
### Summary
This project focus on the classification of disaster response.
The data set contain real messages that were sent during disaster events.
I create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

### File structure
<pre>
├── app
│   ├── run.py                -->flask main file
│   └── templates              -->web template
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv    -->original categories data
│   ├── disaster_messages.csv     -->original messages data
│   ├── DisasterResponse.db	      -->database after processing from ETL pipline
│   └── process_data.py         -->ETL pipeline
├── models
│   ├── classifier.pkl          -->meachine learning model    
│   └── train_classifier.py      -->meachine learning pipeline
└── README.md
</pre>
### Prerequisites
- nltk 3.2
- python 3.6
- pandas 0.2
- sklearn 0.2
- plotly 2.0
- flask 0.12
### Run the project
you are welcome to run this project in you own computer,you need to follow these steps:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

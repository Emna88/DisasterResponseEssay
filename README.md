# Disaster Response Pipeline Project

The purpose of this project is a machine learning classifier that reads messages collecting during a disaster and classifies them into categories so they can be answered appropriately.


The proposed algorithm reads messages, clean and tokenize them and fed them to a ML pipeline that permits to classify each message to the appropriate label.

### File Descriptions
data/process_data.py - The ETL script
models/train_classifier - The model building steps
app/run.py - The server for the website
app/templates - The website HTML files
data/*.csv - The dataset

### Installation
Run pip install -r requirements.txt

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

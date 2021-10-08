# Disaster Response Pipelines


### Table of Contents

1. [Installation](#installation)
2. [Data Pipelines](#Data_Pipelines)
3. [Files Descriptions](#files)
4. [Instructions](#instructions)

## Installation <a name="installation"></a>

The requirements of the app:
- python 3.6
- pandas
- numpy
- re
- sklearn
- sys
- nltk
- sqlalchemy
- pickle
- Flask
- plotly



## Data Pipelines<a name = "Data_Pipelines"></a>
The project has three componants which are:

1. **ETL Pipeline:** `process_data.py` file contain the script to create ETL pipline which:
The first part of your data pipeline is the Extract, Transform, and Load process.
- Loads the `messages` and `categories` csv files
- Merge them into one dataset
- Cleans the data
- Stores it in a SQL database

2. **ML Pipeline:** `train_classifier.py` file contain the script to create ML pipline which:

- Loads data from the SQL database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Train the models
- Outputs results on the test set
- Exports the final model as a pickle file

3. **Flask Web App:** the web app enables the user to enter a disaster message, and then view the categories of the message.

The web app also contains some visualizations that describe the data. 
 
 
 
## Files Descriptions <a name="files"></a>

The files structure is arranged as below:

	- README.md: read me file
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
	- workspace
		- \app
			- run.py: flask file to run the app
		- \templates
			- master.html: main page of the web application 
			- go.html: result web page
		- \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- DisasterResponse.db: disaster response database
			- process_data.py: ETL process
		- \models
			- train_classifier.py: classification code

## Instructions <a name="instructions"></a>

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

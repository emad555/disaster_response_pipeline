import sys

import numpy as np
import pandas as pd

#used sqlalchemy for sql commands.
from sqlalchemy import create_engine

# load data is used to load the data from disaster_categories.csv and disaster_messages and merg
# them togather in one dataframe.

def load_data(messages_filepath, categories_filepath):
    '''
    This function reads the data from the csv files and merge them togather based on the id column.
    parmeters: 
    
    messages_filepath (Str): the path of the csv file of messages dataset
    categories_filepath (Str): the path of the csv file of categories dataset
   
    Returns the two datasets compined in one dataset.
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge both files in one dataset based on the id
    df = messages.merge(categories, how='outer',on=['id'])
    return df

def clean_data(df):   
    
    '''
    This function used to clean the data by converting the category values into 0 and 1.
    Then separate the catigories and save them in an array with the name new_categories.
    The function also convert the catigories values into binary numbers and drop the values that are nor 1 or 0. 
    Then print the numbers of zeros and ones for each category in the terminal.
    Lastly add the array's elements to the converted data as headers and merge the new columns to the old dataset.
    
    df (pandas.core.frame.DataFrame): the dataset provided from load_data.
    
    Returns (dropped_duplicates) dataset with category columns filled by only 1 or 0 and without duplicates.
    
    '''
    
#     df = load_data('disaster_messages.csv' , 'disaster_categories.csv')
    splitted_categories = df['categories'].str.split(';', expand=True)
    # categories = categories['categories'].str.split(expand=True)
    for i in range(36):
        splitted_categories[i] = splitted_categories[i].str.replace('\D', '')
    
    replaced = df['categories'].iloc[0].split(";")
    new_categories = []
    for i in range(int(len(replaced))):
        replaced = df['categories'].iloc[0].split(";")[i].replace('-1','').replace('-0','').replace('_',' ')
        stri = str(replaced)
        new_categories.append(replaced)
    print(new_categories)

    new = pd.DataFrame(columns=new_categories)
    

    for i in range(36):
        splitted_categories.rename(columns={i:new_categories[i]}, inplace=True)

    merged = [df, splitted_categories]
    result = pd.concat(merged, axis=1)
    for i in new_categories:
        result.drop(result.index[result[i] == 2], inplace = True)
    dropped_duplicates = result.drop_duplicates()
    
    for i in new_categories:
        print(result[i].value_counts())
    
    return dropped_duplicates
    
    
def save_data(df, database_filename):
    '''
    This function used to covert the data into SQL data and save them in spcifice path.
    
    df (pandas.core.frame.DataFrame): the dataset provided from clean_data.
    database_filename: the path of the dataset.
    '''
    dropped_duplicates = clean_data(df)
    engine = create_engine('sqlite:///DisasterResponse.db')
    dropped_duplicates.to_sql('DisasterResponse', engine, index=False,if_exists='replace')  
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
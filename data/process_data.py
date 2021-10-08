import sys

import numpy as np
import pandas as pd

#used sqlalchemy for sql commands.
from sqlalchemy import create_engine

# load data is used to load the data from disaster_categories.csv and disaster_messages and merg
# them togather in one dataframe.

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge both files in one dataset based on the id
    df = messages.merge(categories, how='outer',on=['id'])
    return df

def clean_data(df):
    df = load_data('disaster_messages.csv' , 'disaster_categories.csv')
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
    dropped_duplicates = result.drop_duplicates()
    return dropped_duplicates
    
    
def save_data(df, database_filename):
    dropped_duplicates = clean_data(df)
    engine = create_engine('sqlite:///DisasterResponse.db')
    dropped_duplicates.to_sql('DisasterResponse', engine, index=False)  


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
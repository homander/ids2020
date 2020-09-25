import numpy as np
import pandas as pd


# Drops columns from dataframes
def drop_columns(data, *args):
    for i in args:
        data.drop(columns=data.columns[i], inplace=True)


# Encodes missing values as np.nan
def encode_values(data):
    data.replace([-1, '-1', 'Unknown'], [np.nan, np.nan, np.nan], inplace=True)


###################
# Import the data #
###################

# Import the CSV file
df_all_jobs = pd.read_csv('../data/all_jobs.zip')
df_scientist = pd.read_csv('../data/data-scientist-jobs.zip')
df_engineer = pd.read_csv('../data/data-engineer-jobs.zip')
df_glassdoor = pd.read_csv('../data/glassdoor-data-science-jobs.zip')

#############################
# The 1st phase of cleaning #
#############################

# Drop columns that we probably won't use
# and which have a lot of missing values
drop_columns(df_all_jobs, 16, 15, 14, 1, 0)  # drop 'Unnamed: 0', 'Unnamed: 1, 'Revenue', 'Competitors', 'Easy Apply'
drop_columns(df_scientist, 16, 15, 14, 1, 0)  # drop 'Unnamed: 0', 'index', 'Revenue', 'Competitors', 'Easy Apply'
drop_columns(df_engineer, 14, 13, 12)  # drop 'Revenue', 'Competitors', 'Easy Apply'
drop_columns(df_glassdoor, 13, 12)  # drop 'Revenue', 'Competitors'

##############################
# Encoding missing values    #
##############################

# Encode missing values as np.nan
encode_values(df_all_jobs)
encode_values(df_scientist)
encode_values(df_engineer)
encode_values(df_glassdoor)

# Concatenate datasets
df_full = pd.concat([df_all_jobs, df_scientist, df_engineer, df_glassdoor], ignore_index=True)

# Drop duplicates
df_full = df_full.drop_duplicates().reset_index(drop=True)

# Create dataframe without nan values
df = df_full.dropna(axis=0, how='any').reset_index(drop=True)

##############################
# Save the preprocessed data #
##############################

# Save the preprocessed data
# Save also as an Excel workbook for easier viewing

df_all_jobs.to_csv('../processed-data/all_jobs.csv')
df_all_jobs.to_excel('../processed-data/all_jobs.xlsx')

df_scientist.to_csv('../processed-data/data_scientist.csv')
df_scientist.to_excel('../processed-data/data_scientist.xlsx')

df_engineer.to_csv('../processed-data/data_engineer.csv')
df_engineer.to_excel('../processed-data/data_engineer.xlsx')

df_glassdoor.to_csv('../processed-data/glassdoor_jobs.csv')
df_glassdoor.to_excel('../processed-data/glassdoor_jobs.xlsx')

df_full.to_csv('../processed-data/df_full.csv')
df_full.to_excel('../processed-data/df_full.xlsx')

df.to_csv('../processed-data/df.csv')
df.to_excel('../processed-data/df.xlsx')

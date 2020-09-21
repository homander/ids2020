import numpy as np
import pandas as pd

###################
# Import the data #
###################

# Import the CSV file
df = pd.read_csv('data/DataScientist.csv')

#############################
# The 1st phase of cleaning #
#############################

# Drop the index columns
df.drop(columns=df.columns[:2], inplace=True)

# Drop columns that we probably won't use
# and which have a lot of missing values
df.drop(columns=['Revenue', 'Competitors', 'Easy Apply'], inplace=True)

# Encode missing values as np.nan
df.replace([-1, '-1', 'Unknown'], [np.nan, np.nan, np.nan], inplace=True)

##############################
# Save the preprocessed data #
##############################

# Save the preprocessed data
# Save also as an Excel workbook for easier viewing
df.to_csv('processed-data/DataScientist_preprocessed.csv')
df.to_excel('processed-data/DataScientist_preprocessed.xlsx')

print(df)

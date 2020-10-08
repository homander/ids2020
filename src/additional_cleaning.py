import pandas as pd
import re

# Import preprocessed CSV file
df = pd.read_csv('processed-data/df_cleaned.csv', index_col=0)

############################
# ADDITIONAL DATA CLEANING #
############################

# remove rating from Company name
df['Company Name'] = df["Company Name"].str.partition("\n")

# Separate City and State location
df['Location City'] = df['Location'].str.split(",", expand=True)[0]
df['Location State'] = df['Location'].str.split(", ", expand=True)[1]
df['Headquarters City'] = df['Headquarters'].str.split(",", expand=True)[0]
df['Headquarters State'] = df['Headquarters'].str.split(", ", expand=True)[1]

# Drop processed columns and rows with NAN
df = df.dropna(axis=0, how='any')
df.drop('Location', axis=1, inplace=True)
df.drop('Headquarters', axis=1, inplace=True)
df = df[df['Location State'].map(len) == 2]
df = df[df['Headquarters State'].map(len) == 2].reset_index(drop=True)

# Clean Job Title
df['Job Title'] = df['Job Title'].apply(lambda x: re.sub(r'(?i)(sr\.)|sr|lead', 'Senior', x))
df['Job Title'] = df['Job Title'].apply(lambda x: re.sub(r'(?i)(jr\.)|jr|â Junior', 'Junior', x))

# Save to files for visualisation
df.to_csv('processed-data/df_cleaned_for_visualisation.csv')
df.to_excel('processed-data/df_cleaned_for_visualisation.xlsx')

import numpy as np
import pandas as pd
import re


# Extracts numbers from 'Salary Estimate'
def extract_numbers(x, y):
    return list(map(float, re.findall(r"(\d+)", x)))[y]


# Import preprocessed CSV file
df = pd.read_csv('processed-data/df.csv')

#######################################
# Add columns with min and max salary #
#######################################


# Define number of working hours per year in US for the cases with 'Per hour' salary data
hours = 1768

# Extract salary info from 'Salary Estimate':
# 1. Check condition on hour/year salary base
# 2. Extract min and max
# 3. For hourly wage: multiply by hours number per year
# 4. For year salary: multiply by 1000
# 5. Add columns 'Min Salary' and 'Max Salary'

# Min is 0-st element in list (args=(0,))
df['Min Salary'] = np.where((df['Salary Estimate'].str.contains("Hour")),
                            (hours * df['Salary Estimate'].apply(extract_numbers, args=(0,)) / 1000).astype(int) * 1000,
                            df['Salary Estimate'].apply(extract_numbers, args=(0,)) * 1000)
# Max is 1-st element in list (args=(1,))
df['Max Salary'] = np.where((df['Salary Estimate'].str.contains("Hour")),
                            (hours * df['Salary Estimate'].apply(extract_numbers, args=(1,)) / 1000).astype(int) * 1000,
                            df['Salary Estimate'].apply(extract_numbers, args=(1,)) * 1000)

# Add column 'Avg Salary'
df['Avg Salary'] = (df['Max Salary'] + df['Min Salary']) / 2

##############
# Clean data #
##############


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

# Drop 'Salary Estimate' column
df.drop(columns=df.columns[[2, 0]], inplace=True)  # drop 'Unnamed: 0', 'Salary Estimate'

# Save to csv, excel file
df.to_csv('processed-data/df_cleaned.csv')
df.to_excel('processed-data/df_cleaned.xlsx')

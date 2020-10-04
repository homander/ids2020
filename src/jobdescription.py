import numpy as np
import pandas as pd
import re

#######################################################
# Extract from Job Description 'skills required'      #
#######################################################


# Function applies mask to job description
def mask(x):
    if re.search(trigger_mask, x):
        return re.search(trigger_mask, x).group()
    else:
        return np.nan


# Import preprocessed CSV file
df = pd.read_csv('processed-data/df_cleaning_2.csv', index_col=0)

#########################################################################
# Form a table with statistics of trigger words frequency in Job Title #
#########################################################################

# List of job titles we interested in
job_titles = ['data scientist', 'data engineer', 'data analyst']

# Create a list of trigger words
trigger_words = ['experience', 'qualification', 'skills', 'proficienc', 'competenc',
                 'requirements', 'about you', 'must have', 'area of expertise']

# Create dataframe with job titles (columns) and trigger words (rows)
triggers = pd.DataFrame(columns=[job_titles[0], job_titles[1], job_titles[2]],
                        index=['# number of jobs #', 'experience', 'qualification', 'skills', 'proficiency',
                               'competence', 'requirements', 'about you', 'must have', 'area of expertise'])

# Extract job description by job titles to temp series
data_scientist_description = df['Job Description'][df['Job Title'].str.contains(f'(?i){job_titles[0]}')]
data_engineer_description = df['Job Description'][df['Job Title'].str.contains(f'(?i){job_titles[1]}')]
data_analyst_description = df['Job Description'][df['Job Title'].str.contains(f'(?i){job_titles[2]}')]

# Filling the table of number trigger words by job titles
triggers['data scientist'].iloc[0] = data_scientist_description.count()
triggers['data engineer'].iloc[0] = data_engineer_description.count()
triggers['data analyst'].iloc[0] = data_analyst_description.count()

for i in range(1, len(trigger_words) + 1):
    triggers['data scientist'].iloc[i] = data_scientist_description[
        data_scientist_description.str.contains(f'(?i){trigger_words[i - 1]}')].count()
    triggers['data engineer'].iloc[i] = data_engineer_description[
        data_engineer_description.str.contains(f'(?i){trigger_words[i - 1]}')].count()
    triggers['data analyst'].iloc[i] = data_analyst_description[
        data_analyst_description.str.contains(f'(?i){trigger_words[i - 1]}')].count()

print('Table with trigger words for Job description cleaning\n', triggers)

##############################################################
# Extract text with skill required part from Job Description #
##############################################################

# Create a mask for text extraction
trigger_mask = f'(?i)({trigger_words[0]}|{trigger_words[1]}|{trigger_words[2]}|{trigger_words[3]}|{trigger_words[4]}|{trigger_words[5]}|{trigger_words[6]}|{trigger_words[7]}|{trigger_words[8]})((?:[^\n][\n]?)+)'

# Clean Job description from unnecessary symbols
df['Job Description'] = df['Job Description'].apply(lambda x: re.sub('^\s+|\u2022|\n|\r|\s+$', ' ', x))

# Apply mask (results saved to new column of dataframe
df['Skills required'] = df['Job Description'].apply(mask)

######################################################
# Count statistics: before text extraction and after #
######################################################

# Count statistics 'before'

# Average length of job description
df['Length'] = df['Job Description'].str.split(' ').str.len()
# Average length of job description with Job title 'Data Scientist'
df['Length_scientist'] = df['Job Description'][df['Job Title'].str.contains(f'(?i){job_titles[0]}')].str.split(
    ' ').str.len()
# Average length of job description with Job title 'Data Engineer'
df['Length_engineer'] = df['Job Description'][df['Job Title'].str.contains(f'(?i){job_titles[1]}')].str.split(
    ' ').str.len()
# Average length of job description with Job title 'Data Analyst'
df['Length_analyst'] = df['Job Description'][df['Job Title'].str.contains(f'(?i){job_titles[2]}')].str.split(
    ' ').str.len()

# Print 'before' statistics
print(f"\nAvg.words in ad before cleaning Job Description:\n"
      f"Scientist- {round(df['Length_scientist'].mean())}, "
      f"Engineer- {round(df['Length_engineer'].mean())}, "
      f"Analyst- {round(df['Length_analyst'].mean())}, "
      f"Total 21685 ads - {round(df['Length'].mean())}")

# Count statistics 'After'

# Average length of skills required
df['Length2'] = df['Skills required'].str.split(' ').str.len()
# Average length of skills required with Job title 'Data Scientist'
df['Length_scientist2'] = df['Skills required'][df['Job Title'].str.contains(f'(?i){job_titles[0]}')].str.split(
    ' ').str.len()
# Average length of skills required with Job title 'Data Engineer'
df['Length_engineer2'] = df['Skills required'][df['Job Title'].str.contains(f'(?i){job_titles[1]}')].str.split(
    ' ').str.len()
# Average length of skills required with Job title 'Data Analyst'
df['Length_analyst2'] = df['Skills required'][df['Job Title'].str.contains(f'(?i){job_titles[2]}')].str.split(
    ' ').str.len()

# Number of job postings wasn't covered by mask
number_total = df['Skills required'].isna().sum()
number_data_scientist = df['Skills required'][df['Job Title'].str.contains(f'(?i){job_titles[0]}')].isna().sum()
number_data_engineer = df['Skills required'][df['Job Title'].str.contains(f'(?i){job_titles[1]}')].isna().sum()
number_data_analyst = df['Skills required'][df['Job Title'].str.contains(f'(?i){job_titles[2]}')].isna().sum()

# Print 'after' statistics
print(f"\nAvg.words in ad after cleaning Job Description:\n"
      f"Scientist- {round(df['Length_scientist2'].mean())}, "
      f"Engineer- {round(df['Length_engineer2'].mean())}, "
      f"Analyst- {round(df['Length_analyst2'].mean())} "
      f"Total 21419 ads - {round(df['Length2'].mean())},\n"
      f"\nAverage % of length decrease- {round(df['Length2'].mean()) / round(df['Length'].mean()):.2%},\n"
      f"Entries without 'skills required' after cleaning- {number_total} ({number_total / (triggers['data scientist'].iloc[0] + triggers['data engineer'].iloc[0] + triggers['data analyst'].iloc[0]):.2%}), "
      f"scientists - {number_data_scientist} ({number_data_scientist / triggers['data scientist'].iloc[0]:.2%}), "
      f"engineers - {number_data_engineer} ({number_data_engineer / triggers['data engineer'].iloc[0]:.2%}), "
      f"analytics - {number_data_analyst} ({number_data_analyst / triggers['data analyst'].iloc[0]:.2%}), ")

# Drop temporary columns:
df.drop(columns=df.columns[[23, 22, 21, 20, 19, 18, 17, 16]], inplace=True)

# Create new dataframe without nan values in Skills required column (loosing 1-2% of entries)
df_cleaned = df.dropna(axis=0, how='any').reset_index(drop=True)

# Save to CSV, excel file
df_cleaned.to_csv('processed-data/df_cleaned.csv')
df_cleaned.to_excel('processed-data/df_cleaned.xlsx')

df.to_csv('processed-data/df_cleaned_nan.csv')
df.to_excel('processed-data/df_cleaned_nan.xlsx')

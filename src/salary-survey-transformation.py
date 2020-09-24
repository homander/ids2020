# A job to transform salary survey input data

import pandas as pd
import string
import country_converter as cc

df = pd.read_csv('data/2019_Data_Professional_Salary_Survey_Responses.zip', compression='zip', encoding='cp850')

# Drop unused columns
df.drop(['Timestamp', 'PostalCode', 'YearsWithThisDatabase', 'EmploymentStatus', 'HoursWorkedPerWeek', 'TelecommuteDaysPerWeek',
         'PopulationOfLargestCityWithin20Miles','LookingForAnotherJob', 'CareerPlansThisYear', 'Gender', 'Counter', 
         'ManageStaff', 'CompanyEmployeesOverall', 'HowManyCompanies', 'OtherPeopleOnYourTeam','DatabaseServers'], axis=1, inplace=True)

# rename columns
df.rename(columns={'Survey Year': 'year', 'Country': 'country_name', 'JobTitle': 'job_title', 
                   'YearsWithThisTypeOfJob': 'experience', 'EmploymentSector': 'sector',
                   'SalaryUSD' : 'salary_yearly'}, inplace=True)


# Combine skills from columns: PrimaryDatabase, OtherDatabases
df['skills'] = df['PrimaryDatabase'].str.cat(df['OtherDatabases'], sep=',', na_rep='')
df['skills']= df['skills'].str.rstrip(',')
df.drop(['PrimaryDatabase', 'OtherDatabases'], axis=1, inplace=True)

# Take only job titles 'Data scientist', 'Engineer' and 'Analyst' for now
job_title_filter = (df['job_title'] == 'Data Scientist') | (df['job_title'] == 'Engineer') | (df['job_title'] == 'Analyst')
df = df[job_title_filter]

# salary_yearly: remove currency symbols and decimal parts and convert to int
df['salary_yearly'] = df['salary_yearly'].replace('\.\d\d\Z', '', regex=True) # remove last two decimals
punct = string.punctuation + '$'
transtab = str.maketrans(dict.fromkeys(string.punctuation, ''))
df['salary_yearly'] = df['salary_yearly'].str.translate(transtab)
df['salary_yearly'] = df['salary_yearly'].astype('int64')

# Process countries
# Drop rows with troublesome identified country names
indexes = df[ df['country_name'] == 'Serbia and Montenegro' ].index
df.drop(indexes, inplace=True)
df.reset_index(drop=True, inplace=True)

numeric_codes = cc.convert(names = df['country_name'].to_list(), to = 'ISOnumeric')
df['country'] = pd.Series(numeric_codes, dtype='int64')

#process sectors
#only two students in data set, will map them to private sector which is more prominent in data
sector_mapping = {'Private business': 2, 
                  'Local government': 1,
                  'State/province government': 1,
                  'Federal government': 1,
                  'Student': 1,
                  'Non-profit': 3,
                  'Education (K-12, college, university)': 1}
df['sector'] = df['sector'].map(sector_mapping)
df['sector'].astype('int32')

#record type
df['type'] = 2

#save preprocessed data
df.to_csv('processed-data/salarysurvey_preprocessed.csv')





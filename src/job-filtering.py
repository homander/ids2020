import pandas as pd
import sys

# Filters jobs by their title to include only jobs which
# are similar to data scientist/engineer/analyst.

# Whether the title should be an exact match, e.g. "data scientist".
# Or, should we include approximate matches, such as "data scientist, analytics"
# A value between 0 (most strict) and 2 (least strict)
FILTER_LEVEL = 1

if len(sys.argv) != 2:
    print('Wrong number of arguments.')
    exit()

file_name = sys.argv[1]

df = pd.read_csv('processed-data/' + file_name, index_col=0)
print(f'Found {df.shape[0]} jobs, out of which')

if FILTER_LEVEL == 0:
    # Require an exact match
    scientist = df['Job Title'].str.lower() == 'data scientist'
    engineer = df['Job Title'].str.lower() == 'data engineer'
    analyst = df['Job Title'].str.lower() == 'data analyst'
elif FILTER_LEVEL == 1:
    # Accept titles which include, for instance, the string "data scientist"
    # This catches titles such as "senior data scientist" and "NLP data scientist"
    scientist = df['Job Title'].str.contains('data scientist', case=False)
    engineer = df['Job Title'].str.contains('data engineer', case=False)
    analyst = df['Job Title'].str.contains('data analyst', case=False)
else:
    # Accept titles which include, for instance, the strings "data" and "scientist",
    # but not necessarily in that order. This catches titles such as "analyst, data"
    # Also, allow "machine learning" instead of "data".
    # In other words, include jobs where the title matches both
    # (data|machine learning) and (scientist|engineer|analyst)
    data = (df['Job Title'].str.contains('data', case=False) |
            df['Job Title'].str.contains('machine learning', case=False))

    scientist = data & df['Job Title'].str.contains('scientist', case=False)
    engineer = data & df['Job Title'].str.contains('engineer', case=False)
    analyst = data & df['Job Title'].str.contains('analyst', case=False)

# Add "Job Type" column
# Done in reverse order because some jobs match more than one category
# So, for instance, jobs which match both "data scientist" and "data engineer"
# get classified as "data scientist" jobs
df.loc[analyst, 'Job Type'] = 'data analyst'
df.loc[engineer, 'Job Type'] = 'data engineer'
df.loc[scientist, 'Job Type'] = 'data scientist'

# Drop jobs which don't match any of the specified types
df = df[scientist | engineer | analyst]

# Sort by job type
df.sort_values(by='Job Type', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# Print statistics (after accounting for multiple matches)
scientist2 = df['Job Type'] == 'data scientist'
engineer2 = df['Job Type'] == 'data engineer'
analyst2 = df['Job Type'] == 'data analyst'
print(f'...{scientist2.sum()} match "data scientist"')
print(f'...{engineer2.sum()} match "data engineer"')
print(f'...{analyst2.sum()} match "data analyst"')
print(f'at filter level {FILTER_LEVEL}')
print(f'A total of {df.shape[0]} jobs remain')

file_name = file_name[0: file_name.index('.')]

df.to_csv('processed-data/' + file_name + '_filtered''.csv')
df.to_excel('processed-data/' + file_name + '_filtered.xlsx')

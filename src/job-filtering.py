import pandas as pd

# Filters jobs by their title to include only jobs which
# are similar to data scientist/engineer/analyst.

# Whether the title should be an exact match, e.g. "data scientist".
# Or, should we include approximate matches, such as "data scientist, analytics"
# A value between 0 (most strict) and 2 (least strict, default)
FILTER_LEVEL = 2

df = pd.read_csv('processed-data/df.csv', index_col=0)
print(f'Found {df.shape[0]} jobs, out of which')

if FILTER_LEVEL == 0:
    # Require an exact match
    df_scientist = df[df['Job Title'].str.lower() == 'data scientist']
    df_engineer = df[df['Job Title'].str.lower() == 'data engineer']
    df_analyst = df[df['Job Title'].str.lower() == 'data analyst']
elif FILTER_LEVEL == 1:
    # Accept titles which include, for instance, the string "data scientist"
    # This catches titles such as "senior data scientist" and "NLP data scientist"
    df_scientist = df[df['Job Title'].str.contains('data scientist', case=False)]
    df_engineer = df[df['Job Title'].str.contains('data engineer', case=False)]
    df_analyst = df[df['Job Title'].str.contains('data analyst', case=False)]
else:
    # Accept titles which include, for instance, the strings "data" and "scientist",
    # but not necessarily in that order. This catches titles such as "analyst, data"
    # Also, allow "machine learning" instead of "data".
    # In other words, include jobs where the title matches both
    # (data|machine learning) and (scientist|engineer|analyst)
    df_data = df[df['Job Title'].str.contains('data', case=False) |
                 df['Job Title'].str.contains('machine learning', case=False)]

    df_scientist = df_data[df_data['Job Title'].str.contains('scientist', case=False)]
    df_engineer = df_data[df_data['Job Title'].str.contains('engineer', case=False)]
    df_analyst = df_data[df_data['Job Title'].str.contains('analyst', case=False)]

print(f'...{df_scientist.shape[0]} match "data scientist"')
print(f'...{df_engineer.shape[0]} match "data engineer"')
print(f'...{df_analyst.shape[0]} match "data analyst"')
print(f'at filter level {FILTER_LEVEL}')

# Concatenate and remove duplicates (jobs which matched more than one category)
df = pd.concat([df_scientist, df_engineer, df_analyst], ignore_index=True)
df.drop_duplicates(keep='first', inplace=True, ignore_index=True)
print(f'After discarding duplicates, {df.shape[0]} jobs remain (in total)')

df.to_csv('processed-data/df_filtered.csv')
df.to_excel('processed-data/df_filtered.xlsx')

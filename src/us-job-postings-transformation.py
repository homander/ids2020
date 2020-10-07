# A job to transform data-scientist-job-postings-from-the-us.zip dataset
import pandas as pd

df = pd.read_csv('data/data-scientist-job-postings-from-the-us.zip', compression='zip', encoding='utf-8')

# Drop unused columns
df.drop(['crawl_timestamp', 'url', 'category', 'company_name', 'city', 'state',
         'inferred_city','inferred_state', 'inferred_country', 'post_date', 'salary_offered', 'job_board',
         'geo', 'cursor', 'contact_email', 'contact_phone_number','uniq_id','html_job_description'], axis=1, inplace=True)

# rename columns
df.rename(columns={'job_title': 'Job Title', 'job_description':'Job Description'}, inplace=True)

#save preprocessed data
df.to_csv('processed-data/data-scientist-job-postings-from-us-2019-preprocessed.csv')
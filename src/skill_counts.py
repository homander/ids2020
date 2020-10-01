import pandas as pd
import regex as re

jobs = pd.read_csv('processed-data\df_cleaned.csv')

# rename column and remove punctuation - these to the cleanup job instead of here?
jobs.rename(columns={'Skills required': 'description'}, inplace=True)
jobs["description"] = jobs['description'].str.replace(r'[^\w\s]+', ' ')


def load_skills(path):
    file = open(path, "r")
    try:
        content = file.read()
        skills = content.split(",")
    finally:
        file.close()

    return skills


def calculate_skills(data, skill_list, skill_name):
    skill_counts = []
    for skill in skill_list:
        expr = '\\b' + skill + '\\b'
        result_set = data[data['description'].str.contains(expr, flags=re.IGNORECASE, regex=True)]
        count = len(result_set.index)
        skill_counts.append([skill, count, (count / len(data.index)) * 100])

    result = pd.DataFrame(skill_counts, columns=[skill_name, 'Count', 'Percentage'])
    result.sort_values(by=['Count'], ascending=False, inplace=True)

    return result


langs_list = load_skills('resources/skills_programming_languages_onegram.txt')
langs_count = calculate_skills(jobs, langs_list, 'Language')

ds_list = load_skills('resources/skills_datastores_onegram.txt')
ds_count = calculate_skills(jobs, ds_list, 'Data Store')

df_list = load_skills('resources/skills_dataformats_onegram.txt')
df_count = calculate_skills(jobs, df_list, 'Data Format')

cp_list = load_skills('resources/skills_cloudproviders_onegram.txt')
cp_count = calculate_skills(jobs, cp_list, 'Cloud Provider')

ga_list = load_skills('resources/skills_general_analytics_onegram.txt')
ga_count = calculate_skills(jobs, ga_list, 'General Data Processing')

ap_list = load_skills('resources/skills_analytics_products_onegram.txt')
ap_count = calculate_skills(jobs, ap_list, 'Analytics Product')

aws_list = load_skills('resources/skills_awsservices_onegram.txt')
aws_count = calculate_skills(jobs, aws_list, 'AWS Service')

# print(langs_count)
# print(ds_count)
# print(df_count)
# print(cp_count)
# print(ga_count)
# print(ap_count)
# print(aws_count)

langs_count.to_csv('processed-data/skill-counts-programming-languages.csv')
ds_count.to_csv('processed-data/skill-counts-data-stores.csv')
df_count.to_csv('processed-data/skill-counts-data_formats.csv')
cp_count.to_csv('processed-data/skill-counts-cloud-providers.csv')
ga_count.to_csv('processed-data/skill-counts-general-analytics.csv')
ap_count.to_csv('processed-data/skill-counts-analytics-products.csv')
aws_count.to_csv('processed-data/skill-counts-aws-services.csv')

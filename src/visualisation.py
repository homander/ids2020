from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import preprocessed CSV file
df = pd.read_csv('processed-data/df_prepared_for_visualisation.csv', index_col=0)

#########################################
# VISUALIZATION PART (will be extended) #
#########################################

df["US"] = "United States"
fig = px.treemap(df, path=['US', 'Location State', 'Location City'], values='Avg Salary')
fig.show()

# Explore Company's rating and average salary correlation
plt.scatter(df['Rating'], df['Avg Salary'])
plt.xlabel("Avg Salary")
plt.ylabel("Company's Rating")
plt.savefig("visualisation/rating_salary_correlation.png")

# Show salary distribution
plt.figure(figsize=(15, 5))
sns.distplot(df['Min Salary'], color="b")
sns.distplot(df['Max Salary'], color="r")
plt.xlabel("Salary in US")
plt.legend({'Min Salary': df['Min Salary'], 'Max Salary': df['Max Salary']})
plt.title("Distribution of Salary in US", fontsize=15)
plt.tight_layout()
plt.savefig("visualisation/salary_distribution.png")

# Most popular job titles
plt.subplots(figsize=(10, 5))
sns.barplot(x=df['Job Title'].value_counts()[0:10].index, y=df['Job Title'].value_counts()[0:10])
plt.xlabel('Job Title', fontsize=5)
plt.ylabel('Job Count', fontsize=10)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Top 10 Job Titles')
plt.savefig("visualisation/most_popular_job_titles.png")

sector_count = pd.DataFrame(df['Sector'].value_counts()[0:10])
industry_count = pd.DataFrame(df['Industry'].value_counts()[0:10])
city_count = pd.DataFrame(df['Location City'].value_counts()[0:10])
ownership_count = pd.DataFrame(df['Type of ownership'].value_counts()[0:5])
company_count = pd.DataFrame(df['Company Name'].value_counts()[0:10])

# Top 5 Types of Ownership of The Companies Searching for DS-employees
plt.style.use('seaborn')
plt.subplots(figsize=(8, 5))
labels = ownership_count.index
sizes = ownership_count['Type of ownership']
pie = plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 5 Types of Ownership of The Companies Searching for DS-employees')
plt.savefig("visualisation/top5_types_ownership.png")

# Top 10 Sectors with Data-related Job Posts
plt.style.use('seaborn')
plt.subplots(figsize=(8, 5))
labels = sector_count.index
sizes = sector_count['Sector']
plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 10 Sectors with Data-related Job Posts')
plt.savefig("visualisation/top10_sectors.png")

# Top 10 Industries Searching for DS-employees
plt.style.use('seaborn')
plt.subplots(figsize=(8, 5))
labels = industry_count.index
sizes = industry_count['Industry']
plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 10 Industries Searching for DS-employees')
plt.savefig("visualisation/top10_industries.png")

# Top 10 Companies Searching for DS-employees
plt.style.use('seaborn')
plt.subplots(figsize=(8, 5))
labels = company_count.index
sizes = company_count['Company Name']
plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 10 Companies Searching for DS-employees')
plt.savefig("visualisation/top10_companies.png")

# Top 10 Cities in US with Data-related Jobs
plt.style.use('seaborn')
plt.subplots(figsize=(8, 5))
labels = city_count.index
sizes = city_count['Location City']
plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 10 Cities in US with Data-related Jobs')
plt.savefig("visualisation/top10_cities.png")

plt.subplots(figsize=(15, 15))
wc = WordCloud()
text = df['Job Title']
wc.generate(str(' '.join(text)))
plt.imshow(wc)
plt.axis("off")
plt.savefig("visualisation/wordcloud_job_titles.png")

plt.subplots(figsize=(15, 15))
wc = WordCloud()
text = df['Skills required']
wc.generate(str(' '.join(text)))
plt.imshow(wc)
plt.axis("off")
plt.savefig("visualisation/wordcloud_skills_required.png")

experience_education = ["bs", "ms", "phd", "full-time", "intern", "bachelor", "remote", "master", "doctorate",
                        "computer science", "data science"]
skills_languages = ["r", "python", "scala", "ruby", "c++", "java", "perl", "ada", "cobol", "javascript", "vba",
                    "typescript",
                    "php", "matlab", "julia", "html", "bash"]
skills_big_cloud_data = ["hadoop", "spark", "impala", "cassandra", "kafka", "hdfs", "hbase", "hive", "aws", "gcp",
                         "azure", "s3", "redshift", "ec2", "lambda", "route s3", "dynamo"]
skills_machine_learning = ["time series", "machine learning", "regression", "stat", "numpy", "pandas",
                           "data visualization",
                           "data analysis", "data cleaning", "deep learning"]

# Top 10 Most Demanded Education Features
vect = CountVectorizer(vocabulary=experience_education)
experience_education_count = pd.Series(np.ravel((vect.fit_transform(df['Skills_lem']).sum(axis=0))),
                                       index=vect.get_feature_names()).sort_values(ascending=False)
plt.subplots(figsize=(10, 5))
sns.barplot(x=experience_education_count[0:10].index, y=experience_education_count[0:10])
plt.xlabel('Education/Experience', fontsize=5)
plt.ylabel('Count', fontsize=10)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Top 10 Most Demanded Education Features')
plt.savefig("visualisation/top10_education.png")

# Top 10 Most Demanded Programming Languages
vect = CountVectorizer(vocabulary=skills_languages)
skills_languages_count = pd.Series(np.ravel((vect.fit_transform(df['Skills_lem']).sum(axis=0))),
                                   index=vect.get_feature_names()).sort_values(ascending=False)
plt.subplots(figsize=(10, 5))
sns.barplot(x=skills_languages_count[0:10].index, y=skills_languages_count[0:10])
plt.xlabel('Languages', fontsize=5)
plt.ylabel('Languages Count', fontsize=10)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Top 10 Most Demanded Programming Languages')
plt.savefig("visualisation/top10_languages.png")

# Top 10 Most Demanded Big Data / Cloud skills
vect = CountVectorizer(vocabulary=skills_big_cloud_data)
skills_big_cloud_data_count = pd.Series(np.ravel((vect.fit_transform(df['Skills_lem']).sum(axis=0))),
                                        index=vect.get_feature_names()).sort_values(ascending=False)
plt.subplots(figsize=(10, 5))
sns.barplot(x=skills_big_cloud_data_count[0:10].index, y=skills_big_cloud_data_count[0:10])
plt.xlabel('Big Data / Cloud skills', fontsize=5)
plt.ylabel('Count', fontsize=10)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Top 10 Most Demanded Big Data / Cloud skills')
plt.savefig("visualisation/top10_big_data_skills.png")

# Top 10 Most Demanded Data Sciences / Machine Learning Skills
vect = CountVectorizer(vocabulary=skills_machine_learning)
skills_machine_learning_count = pd.Series(np.ravel((vect.fit_transform(df['Skills_lem']).sum(axis=0))),
                                          index=vect.get_feature_names()).sort_values(ascending=False)
plt.subplots(figsize=(10, 5))
sns.barplot(x=skills_machine_learning_count[0:10].index, y=skills_machine_learning_count[0:10])
plt.xlabel('Data Sciences / Machine Learning Skills', fontsize=5)
plt.ylabel('Count', fontsize=10)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Top 10 Most Demanded Data Sciences / Machine Learning Skills')
plt.savefig("visualisation/top10_ML_skills.png")

plt.show()

################
# CORRELATIONS #
################

# Check R squared correlation coefficient between Job Type and average salary
X = df['Job Type_le'].values.reshape(-1, 1)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between job type and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between Company's rating and average salary
X = df['Rating'].values.reshape(-1, 1)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print('Correlations')
print(f"The R squared correlation coefficient between Company's rating and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between industry and average salary
X = df['Industry_le'].values.reshape(-1, 1)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between industry and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between sector and average salary
X = df['Sector_le'].values.reshape(-1, 1)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between sector and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between company and average salary
X = df['Company Name_le'].values.reshape(-1, 1)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between company and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between location and average salary
X = df['Location_City_le'].values.reshape(-1, 1)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between location and salary is: {y_model.score(X, y)}")

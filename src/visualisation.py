import plotly.express as px
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams.update({'figure.max_open_warning': 0})

# Import preprocessed CSV file
df = pd.read_csv('processed-data/df_cleaned_filtered.csv', index_col=0)

######################
# VISUALIZATION PART #
######################

df1 = df[['Min Salary', 'Max Salary', 'Avg Salary', 'Job Type', 'Sector', 'Rating']]

# Treemap diagram
df["US"] = "United States"
fig = px.treemap(df, path=['US', 'Location State', 'Location City'], values='Avg Salary')
fig.show()

# SALARY DISTRIBUTION: MIN AND MAX
sns.displot(df1, x='Min Salary', hue='Job Type', bins=20, element="step")
plt.title('Min Salary distribution in US', fontweight="bold")
plt.ylim([0, 1400])
plt.xlim([0, 250000])
plt.tight_layout()
plt.savefig("visualisation/min salary.png")

sns.displot(df1, x='Max Salary', hue='Job Type', bins=20, element="step")
plt.title('Max Salary distribution in US', fontweight="bold")
plt.ylim([0, 1400])
plt.xlim([0, 250000])
plt.tight_layout()
plt.savefig("visualisation/max salary.png")

sns.catplot(x='Job Type', y='Avg Salary', kind="box", data=df1)
plt.title('Average Salary distribution in US', fontweight="bold")
plt.tight_layout()
plt.savefig("visualisation/avg salary.png")

# Explore Company's rating and average salary correlation
g = sns.FacetGrid(df1, hue='Job Type')
g.map(sns.scatterplot, 'Avg Salary', 'Rating', alpha=.6)
g.add_legend()
plt.title('Max Salary distribution in US', fontweight="bold")
plt.title("Company's rating and average salary correlation")
plt.tight_layout()
plt.savefig("visualisation/rating_salary_correlation.png")

# Show salary distribution
plt.figure(figsize=(10, 5))
sns.distplot(df['Min Salary'], color="b")
sns.distplot(df['Max Salary'], color="r")
plt.xlabel("Salary in USD")
plt.legend({'Min Salary': df['Min Salary'], 'Max Salary': df['Max Salary']})
plt.title("Distribution of Salary in US", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("visualisation/salary_distribution.png")

# Most popular job titles
plt.subplots(figsize=(10, 5))
sns.barplot(x=df['Job Title'].value_counts()[0:10].index, y=df['Job Title'].value_counts()[0:10])
plt.xlabel('Job Title', fontsize=10)
plt.ylabel('Job Count', fontsize=10)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Top 10 Job Titles', fontweight="bold")
plt.tight_layout()
plt.savefig("visualisation/most_popular_job_titles.png")

sector_count = pd.DataFrame(df['Sector'].value_counts()[0:10])
industry_count = pd.DataFrame(df['Industry'].value_counts()[0:10])
city_count = pd.DataFrame(df['Location City'].value_counts()[0:10])
ownership_count = pd.DataFrame(df['Type of ownership'].value_counts()[0:5])
company_count = pd.DataFrame(df['Company Name'].value_counts()[0:10])

# Top 5 Types of Ownership of The Companies Searching for DS-employees
plt.subplots(figsize=(8, 5))
labels = ownership_count.index
sizes = ownership_count['Type of ownership']
pie = plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 5 Types of Ownership of The Companies Searching for DS-employees', fontweight="bold")
plt.tight_layout()
plt.savefig("visualisation/top5_types_ownership.png")

# Top 10 Sectors with Data-related Job Posts
plt.figure(figsize=(8, 5))
labels = sector_count.index
sizes = sector_count['Sector']
plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 10 Sectors with Data-related Job Posts', fontweight="bold")
plt.savefig("visualisation/top10_sectors.png")

# Top 10 Industries Searching for DS-employees
plt.figure(figsize=(8, 5))
labels = industry_count.index
sizes = industry_count['Industry']
plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 10 Industries Searching for DS-employees', fontweight="bold")
plt.savefig("visualisation/top10_industries.png")

# Top 10 Companies Searching for DS-employees
plt.figure(figsize=(8, 5))
labels = company_count.index
sizes = company_count['Company Name']
plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 10 Companies Searching for DS-employees', fontweight="bold")
plt.savefig("visualisation/top10_companies.png")

# Top 10 Cities in US with Data-related Jobs
plt.figure(figsize=(8, 5))
labels = city_count.index
sizes = city_count['Location City']
plt.pie(sizes, labels=labels, shadow=True, autopct='%1.1f%%')
plt.title('Top 10 Cities in US with Data-related Jobs', fontweight="bold")
plt.savefig("visualisation/top10_cities.png")

plt.figure(figsize=(8, 5))
wc = WordCloud(background_color='White')
text = df['Job Title']
wc.generate(str(' '.join(text)))
plt.imshow(wc)
plt.axis("off")
plt.savefig("visualisation/wordcloud_job_titles.png")

plt.figure(figsize=(8, 5))
wc = WordCloud(background_color='White')
text = df['Job Description']
wc.generate(str(' '.join(text)))
plt.imshow(wc)
plt.axis("off")
plt.savefig("visualisation/wordcloud_skills_required.png")

###############################
# MERGING SKILLS IN ONE TABLE #
###############################

# Import and concatenating in one table programming languages skills
lang20 = pd.read_csv('processed-data/2020-counts-skill-programminglanguages.csv', index_col=0)
lang20['Year'] = '2020'
lang19 = pd.read_csv('processed-data/2019-counts-skill-programminglanguages.csv', index_col=0)
lang19['Year'] = '2019'
lang = pd.concat([lang20, lang19])
lang['Type'] = 'Programming Languages'

# Import and concatenating in one table general skills
general20 = pd.read_csv('processed-data/2020-counts-skill-generalmisc.csv', index_col=0)
general20['Year'] = '2020'
general19 = pd.read_csv('processed-data/2019-counts-skill-generalmisc.csv', index_col=0)
general19['Year'] = '2019'
general = pd.concat([general20, general19])
general['Type'] = 'General'

# Import and concatenating in one table general analytics skills
generalanalytics20 = pd.read_csv('processed-data/2020-counts-skill-generalanalytics.csv', index_col=0)
generalanalytics20['Year'] = '2020'
generalanalytics19 = pd.read_csv('processed-data/2019-counts-skill-generalanalytics.csv', index_col=0)
generalanalytics19['Year'] = '2019'
generalanalytics = pd.concat([generalanalytics20, generalanalytics19])
generalanalytics['Type'] = 'General Analytics'

# Import and concatenating in one table devops skills
devops20 = pd.read_csv('processed-data/2020-counts-skill-devops.csv', index_col=0)
devops20['Year'] = '2020'
devops19 = pd.read_csv('processed-data/2019-counts-skill-devops.csv', index_col=0)
devops19['Year'] = '2019'
devops = pd.concat([devops20, devops19])
devops['Type'] = 'Devops'

# Import and concatenating in one table datastore skills
datastore20 = pd.read_csv('processed-data/2020-counts-skill-datastores.csv', index_col=0)
datastore20['Year'] = '2020'
datastore19 = pd.read_csv('processed-data/2019-counts-skill-datastores.csv', index_col=0)
datastore19['Year'] = '2019'
datastore = pd.concat([datastore20, datastore19])
datastore['Type'] = 'Datastore'

# Import and concatenating in one table datapipelines skills
datapipelines20 = pd.read_csv('processed-data/2020-counts-skill-datapipelines.csv', index_col=0)
datapipelines20['Year'] = '2020'
datapipelines19 = pd.read_csv('processed-data/2019-counts-skill-datapipelines.csv', index_col=0)
datapipelines19['Year'] = '2019'
datapipelines = pd.concat([datapipelines20, datapipelines19])
datapipelines['Type'] = 'Datapipelines'

# Import and concatenating in one table dataformats skills
dataformats20 = pd.read_csv('processed-data/2020-counts-skill-dataformats.csv', index_col=0)
dataformats20['Year'] = '2020'
dataformats19 = pd.read_csv('processed-data/2019-counts-skill-dataformats.csv', index_col=0)
dataformats19['Year'] = '2019'
dataformats = pd.concat([dataformats20, dataformats19])
dataformats['Type'] = 'Dataformats'

# Import and concatenating in one table cloud skills
cloud20 = pd.read_csv('processed-data/2020-counts-skill-cloudproviders.csv', index_col=0)
cloud20['Year'] = '2020'
cloud19 = pd.read_csv('processed-data/2019-counts-skill-cloudproviders.csv', index_col=0)
cloud19['Year'] = '2019'
cloud = pd.concat([cloud20, cloud19])
cloud['Type'] = 'Cloud'

# Import and concatenating in one table experience skills
experience20 = pd.read_csv('processed-data/2020-counts-experience.csv', index_col=0)
experience20['Year'] = '2020'
experience19 = pd.read_csv('processed-data/2019-counts-experience.csv', index_col=0)
experience19['Year'] = '2019'
experience = pd.concat([experience20, experience19])
experience['Type'] = 'Experience'

# Import and concatenating in one table education skills
education20 = pd.read_csv('processed-data/2020-counts-education.csv', index_col=0)
education20['Year'] = '2020'
education19 = pd.read_csv('processed-data/2019-counts-education.csv', index_col=0)
education19['Year'] = '2019'
education = pd.concat([education20, education19])
education['Type'] = 'Education'

# Form single table of skills required with year, type of skill and type of job info
skills = pd.concat(
    [education, experience, cloud, dataformats, datapipelines, datastore, devops, generalanalytics, general, lang])
skills_a = pd.DataFrame(skills[['Skill', 'Analyst Count', 'Analyst Frequency', 'Year', 'Type']])
skills_a.rename(columns={'Analyst Count': 'Count', 'Analyst Frequency': 'Frequency'}, inplace=True)
skills_a['Job Type'] = 'Data Analyst'
skills_e = pd.DataFrame(skills[['Skill', 'Engineer Count', 'Engineer Frequency', 'Year', 'Type']])
skills_e.rename(columns={'Engineer Count': 'Count', 'Engineer Frequency': 'Frequency'}, inplace=True)
skills_e['Job Type'] = 'Data Engineer'
skills_s = pd.DataFrame(skills[['Skill', 'Scientist Count', 'Scientist Frequency', 'Year', 'Type']])
skills_s.rename(columns={'Scientist Count': 'Count', 'Scientist Frequency': 'Frequency'}, inplace=True)
skills_s['Job Type'] = 'Data Scientist'
skills = pd.concat([skills_a, skills_e, skills_s]).reset_index(drop=True).sort_values(by='Count', ascending=False)

# Save to file skills tables
skills.to_csv('processed-data/skills.csv')
skills.to_excel('processed-data/skills.xlsx')

###############################################
# TRENDS IN SKILLS IN DEMAND (DATA SCIENTIST) #
###############################################

# Programming Languages In Demand: Trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Type'] == 'Programming Languages') & (skills['Job Type'] == 'Data Scientist')]
sns.barplot(x="Skill", y="Frequency", hue="Year", data=data)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Programming Languages In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Programming Languages', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/programming_languages_trends.png")

# General skills In Demand: trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'General')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=80)
plt.yticks(fontsize=10)
plt.title('General Skills In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/general_skills_trends.png")

# General Analytics skills In Demand: Trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'General Analytics')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=80)
plt.yticks(fontsize=10)
plt.title('General Analytics Skills In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/general_analytics_skills_trends.png")

# Devops skills In Demand: Trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'Devops')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Devops Skills In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/devops_skills_trends.png")

# Datastore skills In Demand: Trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'Datastore')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=80)
plt.yticks(fontsize=10)
plt.title('Datastore Skills In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/datastore_skills_trends.png")

# Datapipelines skills In Demand: Trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'Datapipelines')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=10)
plt.yticks(fontsize=10)
plt.title('Datapipelines Skills In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/datapipelines_skills_trends.png")

# Dataformats skills In Demand: trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'Dataformats')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=10)
plt.yticks(fontsize=10)
plt.title('Dataformats Skills In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/dataformats_skills_trends.png")

# Cloud skills In Demand: Trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'Cloud')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=10)
plt.yticks(fontsize=10)
plt.title('Cloud Skills In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/cloud_skills_trends.png")

# Experience In Demand: Trends
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'Experience')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=60)
plt.yticks(fontsize=10)
plt.title('Experience In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/experience_trends.png")

# Education In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Job Type'] == 'Data Scientist') & (skills['Type'] == 'Education')]
sns.barplot(x='Skill', y='Frequency', hue='Year', data=data)
plt.xticks(rotation=10)
plt.yticks(fontsize=10)
plt.title('Education In Demand (Data Scientists): Trends', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/education_trends.png")

##################################
# SKILLS IN DEMAND BY JOB TITLES #
##################################

# Programming Languages In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'Programming Languages')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Programming Languages In Demand', fontweight="bold")
plt.xlabel('Programming Languages', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/programming_languages.png")

# General skills In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'General')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('General Skills In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/general_skills.png")

# General Analytics skills In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'General Analytics')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=80)
plt.yticks(fontsize=10)
plt.title('General Analytics Skills In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/general_analytics_skills.png")

# Devops skills In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'Devops')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Devops Skills In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/devops_skills.png")

# Datastore skills In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'Datastore')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=80)
plt.yticks(fontsize=10)
plt.title('Datastore Skills In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/datastore_skills.png")

# Datapipelines skills In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'Datapipelines')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=10)
plt.yticks(fontsize=10)
plt.title('Datapipelines Skills In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/datapipelines_skills.png")

# Dataformats skills In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'Dataformats')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=10)
plt.yticks(fontsize=10)
plt.title('Dataformats Skills In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/dataformats_skills.png")

# Cloud skills In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'Cloud')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=10)
plt.yticks(fontsize=10)
plt.title('Cloud Skills In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/cloud_skills.png")

# Experience In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'Experience')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=60)
plt.yticks(fontsize=10)
plt.title('Experience In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/experience.png")

# Education In Demand
plt.figure(figsize=(10, 5))
data = skills[(skills['Year'] == '2020') & (skills['Type'] == 'Education')]
sns.barplot(x='Skill', y='Frequency', hue='Job Type', data=data)
plt.xticks(rotation=10)
plt.yticks(fontsize=10)
plt.title('Education In Demand', fontweight="bold")
plt.xlabel('Skills', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.tight_layout()
plt.savefig("visualisation/education.png")

##################
# Skill profiles #
##################

DIM_Y = 2
DIM_X = 3
TOP_N = 10


def plot_skills(job_type, skill_type, k):
    ax = plt.subplot(DIM_Y, DIM_X, k)
    data = skills[
        (skills['Year'] == '2020') &
        (skills['Type'] == skill_type) &
        (skills['Job Type'] == job_type)
        ]
    data = data[:TOP_N]
    sns.barplot(y='Skill', x='Frequency', data=data, ax=ax)
    # plt.yticks(rotation=20)
    plt.xticks(fontsize=10)
    plt.title(skill_type)
    plt.ylabel('', fontsize=10)
    plt.xlabel('Frequency (%)', fontsize=10)
    plt.tight_layout()


# Skill profile for data scientists
plt.figure(figsize=(10, 5))
plot_skills('Data Scientist', 'Programming Languages', 1)
plot_skills('Data Scientist', 'General Analytics', 2)
plot_skills('Data Scientist', 'Cloud', 3)
plot_skills('Data Scientist', 'General', 4)
plot_skills('Data Scientist', 'Experience', 5)
plot_skills('Data Scientist', 'Education', 6)
# plot_skills('Data Scientist', 'Datapipelines', 5)
# plot_skills('Data Scientist', 'Datastore', 6)
plt.savefig("visualisation/skill_profile_data_scientist.png")

# Skill profile for data engineers
plt.figure(figsize=(10, 5))
plot_skills('Data Engineer', 'Programming Languages', 1)
plot_skills('Data Engineer', 'General Analytics', 2)
plot_skills('Data Engineer', 'Cloud', 3)
plot_skills('Data Engineer', 'General', 4)
plot_skills('Data Engineer', 'Experience', 5)
plot_skills('Data Engineer', 'Education', 6)
# plot_skills('Data Engineer', 'Datapipelines', 5)
# plot_skills('Data Engineer', 'Datastore', 6)
plt.savefig("visualisation/skill_profile_data_engineer.png")

# Skill profile for data analysts
plt.figure(figsize=(10, 5))
plot_skills('Data Analyst', 'Programming Languages', 1)
plot_skills('Data Analyst', 'General Analytics', 2)
plot_skills('Data Analyst', 'Cloud', 3)
plot_skills('Data Analyst', 'General', 4)
plot_skills('Data Analyst', 'Experience', 5)
plot_skills('Data Analyst', 'Education', 6)
# plot_skills('Data Analyst', 'Datapipelines', 5)
# plot_skills('Data Analyst', 'Datastore', 6)
plt.savefig("visualisation/skill_profile_data_analyst.png")

plt.show()

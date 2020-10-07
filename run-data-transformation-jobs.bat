del /q /s ".\processed-data\*.csv"
del /q /s ".\processed-data\*.xlsx"
python src/preprocessing_concatenation.py
python src/salary-survey-transformation.py
python src/us-job-postings-transformation.py
python src/job-filtering.py df.csv
python src/job-filtering.py data-scientist-job-postings-from-us-2019-preprocessed.csv
python src/feature_count_extractor.py data-scientist-job-postings-from-us-2019-preprocessed_filtered.csv 2019
python src/feature_count_extractor.py df_filtered.csv 2020
python src/cleaning_salary.py
python src/jobdescription.py
python src/ngram-generation.py
del /q /s ".\processed-data\*.csv"
del /q /s ".\processed-data\*.xlsx"
python src/preprocessing_concatenation.py
python src/salary-survey-transformation.py
python src/job-filtering.py
python src/feature_count_extractor.py analyst
python src/feature_count_extractor.py engineer
python src/feature_count_extractor.py scientist
python src/cleaning_salary.py
python src/jobdescription.py
python src/ngram-generation.py
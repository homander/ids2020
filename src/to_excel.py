import pandas as pd

# The CSV import tool in Excel has trouble with our datasets.
# Hence, it's better to use pandas for reading the CSV files.
# This extremely simple script just reads the CSV file and
# saves it as an Excel worksheet.

# Import the CSV file
df = pd.read_csv("data/DataScientist.csv")

# For convenience, export to a more human-readable format (Excel)
df.to_excel("processed-data/DataScientist.xlsx", index=False)

print('Done!')

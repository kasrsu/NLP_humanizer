import pandas as pd

# Read the CSV file
csv_file_path = '/home/kasr/Acedmics/3rd_year/NLP/COURSEWORK/data_extraction/links6.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the dataframe
print(data.head())

# Count the number of rows before cleaning
before_count = len(data)

# Remove rows that contain "tetw.org"
data_cleaned = data[~data['links'].str.contains("tetw.org", na=False)]

# Count the number of rows after cleaning
after_count = len(data_cleaned)

# Display the counts
print(f"Number of rows before cleaning: {before_count}")
print(f"Number of rows after cleaning: {after_count}")

# Display the first few rows of the cleaned dataframe
print(data_cleaned.head())
# Save the cleaned data to a new CSV file
clean_csv_file_path = '/home/kasr/Acedmics/3rd_year/NLP/COURSEWORK/data_extraction/c_links6.csv'
data_cleaned.to_csv(clean_csv_file_path, index=False)
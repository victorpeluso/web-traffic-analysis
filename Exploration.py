import pandas as pd
import matplotlib  # Use TkAgg backend
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')


# DATA EXPLORATION, CLEANSING, & TRANSFORMATION


# Load the dataset
data = pd.read_csv('/Users/victorpeluso/Desktop/Comprehensive Web Traffic Dataset/website_wata.csv')

# Display the first few rows
print(data.head())

# Check data types and non-null values
print('\n')
print('Data.Information')
print(data.info())

print('\n')
print('Data.Description')
# Summary statistics
print(data.describe())

# Check for missing values
print('\n')
print('Missing Values Report')
print(data.isnull().sum())

# Calculate the Correlation Matrix
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

# Save the correlation matrix as a CSV
correlation_matrix.to_csv('correlation_matrix.csv')

# Convert the correlation matrix to long format
correlation_matrix = correlation_matrix.stack().reset_index()
correlation_matrix.columns = ['Variable 1', 'Variable 2', 'Correlation']
correlation_matrix = correlation_matrix[correlation_matrix['Variable 1'] != correlation_matrix['Variable 2']]
correlation_matrix.to_csv('correlation_long_format.csv', index=False)


# DATA VISUALIZATIONS


# Histogram of Session Duration
data['Session Duration'].hist()
plt.xlabel('Session Duration')
plt.ylabel('Frequency')
plt.title('Histogram of Session Duration')
plt.show()

# Box Plot of Session Duration
data['Session Duration'].plot(kind='box', vert=False)
plt.title('Box Plot of Session Duration')
plt.yticks(rotation=90)
plt.show()

# Distribution of Traffic Sources
data['Traffic Source'].value_counts().plot(kind='bar')
plt.xlabel('Traffic Source')
plt.ylabel('Frequency')
plt.title('Traffic Sources Distribution')
plt.show()

# Scatterplot for Session Duration vs. Conversion Rate
plt.scatter(data['Session Duration'], data['Conversion Rate'])
plt.xlabel('Session Duration')
plt.ylabel('Conversion Rate')
plt.title('Session Duration vs. Conversion Rate')
plt.show()

# Select only numeric columns for Heatmap
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix of the Heatmap
correlation_matrix = numeric_data.corr()

# Set the figure size
plt.figure(figsize=(10, 8))

# Heatmap for Correlation with adjusted formatting
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})

# Rotate the axis labels for better visibility
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.title('Correlation Heatmap')
plt.show()
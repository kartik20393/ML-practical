import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Sample DataFrame
data = {
    'A': [1, 2, None, 4, 5],
    'B': ['X', None, 'Y', 'Z', 'X'],
    'C': [7, 8, 9, None, 11]
}
df = pd.DataFrame(data)
print("DataSet:\n",df) 

# Handling missing values
imputer = SimpleImputer(strategy='mean')
df[['A', 'C']] = imputer.fit_transform(df[['A', 'C']])
print("DataSet after handling Missing Values of A and C Columns:\n",df[['A','C']]) 

df.loc[df['B'].isna(), 'B'] = 'Unknown'  # Use loc to avoid chained assignment warning

# Encoding categorical variables
label_encoder = LabelEncoder()
df['B_encoded'] = label_encoder.fit_transform(df['B'])
print("\nDataSet after handling Missing Values of B Before Label encoding:\n", df['B_encoded'])
encoded_data = OneHotEncoder().fit_transform(df[['B_encoded']]).toarray()
encoded_df = pd.DataFrame(encoded_data, columns=[f'B_{i}' for i in range(encoded_data.shape[1])])
print("\nDataSet after handling Missing Values of B After Label encoding:\n", df['B'])
# Combining DataFrames
df = pd.concat([df, encoded_df], axis=1)
print("DataSet after handling Missing Values of B After one_hot_encoder:\n",df) 
# Feature scaling
scaled_data = StandardScaler().fit_transform(df[['A', 'C']])
scaled_df = pd.DataFrame(scaled_data, columns=['A_scaled', 'C_scaled'])
df = pd.concat([df, scaled_df], axis=1)

# Display final DataFrame
print("Feature Scaling using Standard scaler\n", df)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Generate 100 samples
n_samples = 100

data = {
    
    'hours_study': np.random.randint(3, 8, n_samples),
    'exam_score': np.random.uniform(50, 95, n_samples),
    
}
df = pd.DataFrame(data)
df.to_csv('data/score_data.csv', index=False)
print(f"Generated {len(df)} samples")

# read the data
df = pd.read_csv('data/score_data.csv')
#extract features and target variable
X = df[['hours_study']]
y = df['exam_score']
#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
#take user input for make predictions
hours = float(input("Enter the number of hours studied: "))
prediction = model.predict([[hours]])
print(f"Predicted exam score for {hours} hours of study: {prediction[0]:.2f}")


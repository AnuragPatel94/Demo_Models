# business problem statement: telecome company wants to predict whether a customer will churn or not based on 
# their usage patterns, customer age, and recharge amount.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.svm import SVC
import joblib
np.random.seed(42)

# Generate 1000 samples
n_samples = 1000
data = {
    'age': np.random.randint(25, 60, n_samples),
    'recharge_amount': np.random.randint(150, 600, n_samples),
    'per month reacharge': np.random.poisson(lam=1, size=n_samples)
}
# Create churn label (0 = Not Churn, 1 = Churn)
churn = np.where((data['recharge_amount'] < 300) & (data['per month reacharge'] < 2), 1, 0)
df = pd.DataFrame(data)
df['churn'] = churn
print(df.head())
df.to_csv('data/Telecome_churn_data.csv', index=False)

print(f"Generated {len(df)} samples")

# read the data
df = pd.read_csv('data/Telecome_churn_data.csv')

# extract features and target variable
X = df[['age', 'recharge_amount', 'per month reacharge']]
y = df['churn']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the svc model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'churn_model.pkl')

#test the model
y_pred = model.predict(X_test)

#check the accuracy of the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

#print report
print("Classification Report:")
print(classification_report(y_test, y_pred))



# business problem statement: telecome company wants to predict whether a customer will churn or not based on 
# their usage patterns, customer age, and recharge amount.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.svm import SVC
np.random.seed(42)

# Generate 800 samples
n_samples = 800
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

#test the model
y_pred = model.predict(X_test)

#check the accuracy of the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

#print report
print("Classification Report:")
print(classification_report(y_test, y_pred))

#take user input for make predictions
user_age = float(input("Enter the age: "))
if user_age < 25 or user_age > 60:
    print("Age should be greater than 25 and less than 60.")
    exit()

user_monthly_recharge = float(input("Enter the monthly recharge amount: "))
if user_monthly_recharge < 150 or user_monthly_recharge > 600:
    print("Recharge amount should be between 150 and 600.")
    exit()

user_times_recharged = float(input("Enter the number of times recharged in a month: "))
if user_times_recharged < 0:
    print("Number of times recharged cannot be negative.")
    exit()

#input set into array and make prediction
user_data=np.array([[user_age, user_monthly_recharge, user_times_recharged]])   
prediction = model.predict(user_data)
if prediction[0] == 1:
    print("The customer is likely to churn.")
else:
    print("The customer is not likely to churn.")



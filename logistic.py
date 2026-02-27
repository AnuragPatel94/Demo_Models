# business problem statement: predict the sentiment of customers based on their age, 
# time spent on the website, and the number of items added to the cart.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
np.random.seed(42)

# Generate 500 samples
n_samples = 500
data = {
    'age': np.random.randint(18, 60, n_samples),
    'timespent': np.random.randint(10, 60, n_samples),
    'addtocart': np.random.poisson(lam=1, size=n_samples)
}

# Optional: Create sentiment score (0 = Negative, 1 = Positive)
sentiment = np.where(data['addtocart'] > 1, 1, 0)
df = pd.DataFrame(data)
df['sentiment'] = sentiment
print(df.head())
df.to_csv('data/Ecom_cust_sentiment_data.csv', index=False)
print(f"Generated {len(df)} samples")

# read the data
df = pd.read_csv('data/Ecom_cust_sentiment_data.csv')

# extract features and target variable
X = df[['age', 'timespent', 'addtocart']]
y = df['sentiment']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the linear regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#check the accuracy of the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

#take user input for make predictions
user_age = float(input("Enter the age: "))
if user_age < 18 or user_age > 60:
    print("Age should be greater than 18 and less than 60.")
    exit()
user_timespent = float(input("Enter the time spent: "))
if user_timespent < 10 or user_timespent > 60:
    print("Time spent should be greater than 10 and less than 60.")
    exit()
user_addtocart = float(input("Enter the number of items added to cart: "))

#input set into array and make prediction
user_data=np.array([[user_age, user_timespent, user_addtocart]])

prediction = model.predict(user_data)
if prediction[0] == 1:
    print("likely to purchase")
else:    print("not likely to purchase")





    
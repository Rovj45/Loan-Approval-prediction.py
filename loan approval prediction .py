import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data or loading from CSV)
data = {
    'income': [50000, 70000, 40000, 65000, 30000],
    'credit_score': [720, 800, 650, 750, 580],
    'employment_years': [5, 10, 2, 8, 1],
    'loan_amount': [20000, 15000, 30000, 25000, 10000],
    'loan_status': ['Approved', 'Approved', 'Denied', 'Approved', 'Denied']  # Target variable
}

df = pd.DataFrame(data)  # Convert data to DataFrame

# Separate features (X) and target variable (y)
X = df.drop('loan_status', axis=1)  # Assuming 'loan_status' is the target
y = df['loan_status']

# Preprocess categorical features (if any) using techniques like one-hot encoding
# ... (preprocessing steps here)

# Standardize numerical features (assuming income, loan amount, and employment years are numerical)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Example prediction for a new applicant
new_applicant = {'income': 60000, 'credit_score': 700, 'employment_years': 3, 'loan_amount': 22000}
new_data = pd.DataFrame([new_applicant])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

if prediction[0] == 'Approved':
  print("Loan approval predicted: Approved")
else:
  print("Loan approval predicted: Denied")
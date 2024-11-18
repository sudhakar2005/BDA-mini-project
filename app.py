import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

# Load the dataset
df = pd.read_csv('flight_dataset.csv')

# Inspect the dataset
print(df.head())

# Preprocessing: Ensure all necessary columns are present
# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=["Airline", "Source", "Destination"], drop_first=True)

# Check for missing values and handle them if any
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median())  # Simple median imputation for missing values

# Define features and target variable
X = df.drop("Price", axis=1)  # Assuming 'Price' is the column to predict
y = df["Price"]

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize models
linear_model = LinearRegression()
decision_tree = DecisionTreeRegressor(random_state=42)
random_forest = RandomForestRegressor(random_state=42)

# Train models
linear_model.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_tree = decision_tree.predict(X_test)
y_pred_forest = random_forest.predict(X_test)

# Evaluation function
def evaluate_model(y_test, y_pred, model_name):
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Accuracy (R² Score): {r2:.2f}")

# Evaluate models
evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_tree, "Decision Tree Regressor")
evaluate_model(y_test, y_pred_forest, "Random Forest Regressor")

joblib.dump(random_forest, "random_forest_regressor_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")  # Save feature names
print("Model and feature names saved.")
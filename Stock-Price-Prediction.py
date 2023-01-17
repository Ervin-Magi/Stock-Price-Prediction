import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate made-up stock data
num_days = 365
open_prices = np.random.randint(20, 100, size=num_days)
high_prices = open_prices + np.random.randint(1, 5, size=num_days)
low_prices = open_prices - np.random.randint(1, 5, size=num_days)
close_prices = open_prices + np.random.randint(-3, 3, size=num_days)
adj_close_prices = close_prices + np.random.randint(-3, 3, size=num_days)
days = pd.date_range(start='2023-01-01', periods=num_days)

# Create a DataFrame with the made-up data
data = pd.DataFrame({'Open': open_prices, 'High': high_prices, 'Low': low_prices, 'Close': close_prices, 'Adj Close': adj_close_prices, 'Days': days})

# Split the data into features (X) and labels (y)
X = data[['Open', 'High', 'Low', 'Close']]
y = data['Adj Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Add the predictions to the test set
data_test = X_test
data_test['Adj Close'] = y_test
data_test['Predictions'] = predictions

#Add the date
data_test['Date'] = days[-num_days//5:]

# Print the final DataFrame
data_test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

google_data = pd.read_csv('GOOG.csv')
apple_data = pd.read_csv('AAPL.csv')

apple_data['Date'] = pd.to_datetime(apple_data['Date'])  # Convert 'Date' column to datetime

X = np.array(apple_data['Date'].index).reshape(-1, 1)

X_train, X_test, apple_y_train, apple_y_test = train_test_split(X, apple_data['Close'], test_size=0.2, random_state=42)

apple_model = LinearRegression()
apple_model.fit(X_train, apple_y_train)

apple_y_pred = apple_model.predict(X_test)

google_data['Date'] = pd.to_datetime(google_data['Date'])  

X_train, X_test, google_y_train, google_y_test = train_test_split(X, google_data['Close'], test_size=0.2, random_state=42)

google_model = LinearRegression()
google_model.fit(X_train, google_y_train)

google_y_pred = google_model.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(apple_data['Date'], apple_data['Close'], color='blue', label='Apple - Historical Data')
plt.plot(apple_data['Date'].iloc[X_test.flatten()], apple_y_pred, color='red', label='Apple - Predicted Price')
plt.scatter(apple_data['Date'].iloc[X_test.flatten()], apple_y_pred, color='red', label='Apple - Predicted Price', marker='o')

plt.plot(google_data['Date'], google_data['Close'], color='green', label='Google - Historical Data')
plt.plot(google_data['Date'].iloc[X_test.flatten()], google_y_pred, color='orange', label='Google - Predicted Price')
plt.scatter(google_data['Date'].iloc[X_test.flatten()], google_y_pred, color='orange', label='Google - Predicted Price', marker='o')

plt.title('Stock Price Comparison: Apple vs Google')
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.xticks(rotation=45)  
plt.legend()
plt.grid(True)

plt.show()

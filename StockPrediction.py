import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


aapl = yf.Ticker("AAPL")
hist = aapl.history(period="max")

X = hist[['Open']].values
y = hist[['Close']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_predicted = model(X_test_tensor)
    test_loss = criterion(y_predicted, y_test_tensor)
    y_predicted_np = y_predicted.numpy()  # Convert predictions to NumPy array for sklearn
    y_test_np = y_test_tensor.numpy()     # Convert true values to NumPy array

print(f'Test loss (MSE using PyTorch): {test_loss.item():.4f}')

mse = mean_squared_error(y_test_np, y_predicted_np)
mae = mean_absolute_error(y_test_np, y_predicted_np)
r2 = r2_score(y_test_np, y_predicted_np)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")


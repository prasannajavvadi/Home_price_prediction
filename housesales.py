import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# Read the data
data = pd.read_csv("kc_house_data.csv")

# Display the first few rows and summary statistics
print(data.head())
print(data.describe())

# Visualizations
data['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()

plt.scatter(data.price, data.sqft_living)
plt.title("Price vs Square Feet")
plt.show()

plt.scatter(data.price, data.long)
plt.title("Price vs Location of the area")
plt.show()

plt.scatter(data.price, data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")
plt.show()

plt.scatter(data.bedrooms, data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()

plt.scatter(data.waterfront, data.price)
plt.title("Waterfront vs Price (0 = no waterfront)")
plt.show()

data.floors.value_counts().plot(kind='bar')
plt.show()

plt.scatter(data.floors, data.price)
plt.show()

plt.scatter(data.condition, data.price)
plt.show()

plt.scatter(data.zipcode, data.price)
plt.title("Price by Zipcode")
plt.show()

# Prepare the data for modeling
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'], axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10, random_state=2)

# Train a linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)
print("Linear Regression Score:", reg.score(x_test, y_test))

# Train a gradient boosting regressor model
clf = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='squared_error')
clf.fit(x_train, y_train)
print("Gradient Boosting Regressor Score:", clf.score(x_test, y_test))

# Plot the training and testing scores
test_scores = np.zeros((clf.n_estimators,), dtype=np.float64)
for i, y_pred in enumerate(clf.staged_predict(x_test)):
    test_scores[i] = clf.loss_(y_test, y_pred)

test_scores = np.arange(clf.n_estimators) + 1

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(test_scores, clf.train_score_, 'b-', label='Training set deviance')
plt.plot(test_scores, test_scores, 'r-', label='Test set deviance')
plt.show()

# PCA
pca = PCA()
pca.fit_transform(scale(train1))

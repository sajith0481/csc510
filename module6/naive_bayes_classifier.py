
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

data = {
            'Age': [25, 30, 45, 35, 22, 60, 40, 48, 33, 55],
                'Income': [50000, 60000, 80000, 62000, 40000, 100000, 75000, 85000, 48000, 90000],
                    'Buys': [0, 0, 1, 1, 0, 1, 1, 1, 0, 1]
                    }

df = pd.DataFrame(data)
X = df[['Age', 'Income']]
y = df['Buys']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

results = pd.DataFrame(scaler.inverse_transform(X_test), columns=['Age', 'Income'])
results['Actual'] = y_test.values
results['Predicted'] = y_pred
results['Probability_0'] = y_prob[:, 0]
results['Probability_1'] = y_prob[:, 1]

print("\n--- Naive Bayes Classification Results ---\n")
print(results.to_string(index=False))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
# Load the dataset (replace 'spam.csv' with your dataset file)
data = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')
# Assuming your dataset columns are like: 'text', 'label' (0 for ham, 1 for spam)
X = data['text']
y = data['label']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
# Initialize a Logistic Regression model
model = LogisticRegression()
# Train the model
model.fit(X_train_vect, y_train)
# Predict on the test set
y_pred = model.predict(X_test_vect)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
# Now you can use the trained model to classify new emails
new_emails = ["Get rich quick!", "Hi, how are you?"]
new_emails_vect = vectorizer.transform(new_emails)
new_predictions = model.predict(new_emails_vect)
print("New email predictions:")
for email, prediction in zip(new_emails, new_predictions):
        print(f"Email: {email}, Prediction: {'spam' if prediction == 1 else 'ham'}")
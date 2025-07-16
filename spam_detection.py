import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
print(df.head())

# Preprocess the data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary

# feature extraction (Bag of Words)
cv = CountVectorizer()
X = cv.fit_transform(df['message'])  # Convert messages to a matrix of token counts
y = df['label']  # Labels

# Split and Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# tru a custom message
sample = ["Congratulations! You've won a lottery!"]
vector = cv.transform(sample)
prediction = model.predict(vector)
print("Prediction for sample message:", "Spam" if prediction[0] == 1 else "Ham")



# ⚙️ How Does It Work?
# Collect Data
# You use a dataset of labeled messages.
# Example:

# ham = not spam

# spam = spam

# Text Preprocessing
# Clean the text:

# Lowercase everything

# Remove punctuation and stopwords (like “is”, “the”, “and”)

# Tokenize (split into words)

# Convert Text to Numbers
# Machines don’t understand words, so we convert them to numbers using:

# Bag of Words

# TF-IDF Vectorizer

# Train the Model
# Use a machine learning algorithm (like Naive Bayes) to learn patterns from the training data.

# Predict
# You give a new message to the model, and it predicts whether it’s spam or ham.



# 📊 Example Dataset:
# 📦 Simple Example Flow:

# Message: "You’ve won $1000! Click now!"
# → Clean the text
# → Convert to numbers
# → Model sees it looks like previous spam messages
# → Predicts: SPAM

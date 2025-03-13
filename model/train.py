import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("Spam_SMS.csv", names=["label", "message"], sep=",", header=None)

# Convert labels to binary (ham: 0, spam: 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop any null values
df.dropna(inplace=True)

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))  # Remove stopwords
    return text

df['message'] = df['message'].apply(preprocess_text)

# Feature Extraction using TF-IDF (with n-grams)
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 4))
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Hyperparameter tuning for Random Forest
rf_params = {'n_estimators': [200, 300, 400], 'max_depth': [20, 25, 30]}
rf_grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), rf_params, cv=3)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Hyperparameter tuning for Naïve Bayes
nb_params = {'alpha': [0.01, 0.05, 0.1, 0.5, 1]}
nb_grid = GridSearchCV(MultinomialNB(), nb_params, cv=3)
nb_grid.fit(X_train, y_train)
best_nb = nb_grid.best_estimator_

# Stacking Model
base_models = [('rf', best_rf), ('nb', best_nb)]
meta_model = LogisticRegression(class_weight={0: 1, 1: 2})  # More weight to spam

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True)
stacking_model.fit(X_train, y_train)

# Save Models
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(stacking_model, "model/spam_model.pkl")

# Predictions & Evaluation
y_pred = stacking_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Training Completed! Model Accuracy: {accuracy:.2f}")
print(report)

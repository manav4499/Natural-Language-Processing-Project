import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


stop_words = stopwords.words('english')

# -------------------------------------
# Step 1: Load the Data
# -------------------------------------

df = pd.read_csv("Data5.csv")

df = df[['Body', 'Label']]
df.dropna(inplace=True)

# -------------------------------------
# Step 2: Basic Data Exploration
# -------------------------------------
print("Data Head:\n", df.head())
print("\nData Info:")
print(df.info())
print("\nClass Distribution:")
print(df['Label'].value_counts())
df['length'] = df['Body'].apply(len)
print("\nAverage comment length:", df['length'].mean())

# -------------------------------------
# Step 3: Text Preprocessing
# -------------------------------------
df['Body'] = df['Body'].str.lower().str.strip()

# -------------------------------------
# Step 4: Apply CountVectorizer (Bag of Words)
# -------------------------------------

# Using parameters to filter out extreme common/rare words
count_vect = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)

#This will store the bag of words after removing the common and rare words
X_counts = count_vect.fit_transform(df['Body'])
y = df['Label']

print("\nAfter CountVectorizer:")
print("Shape of X_counts:", X_counts.shape)

#simply returns a list of all the words (features) that the CountVectorizer found in your text data
feature_names = count_vect.get_feature_names_out()
print("Number of features:", len(feature_names))
print("Sample features:", feature_names[:10])

# -------------------------------------
# Step 5: Apply TF-IDF Transformer
# -------------------------------------
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
print("\nAfter TF-IDF:")
print("Shape of X_tfidf:", X_tfidf.shape)

# -------------------------------------
# Step 6: Shuffle the dataset with frac=2
# -------------------------------------

# We'll use replace=True because frac=2 is larger than the population
df_shuffled = df.sample(frac=2, random_state=42, replace=True)
X_tfidf_shuffled = X_tfidf[df_shuffled.index]
y_shuffled = df_shuffled['Label']

# -------------------------------------
# Step 7: Split into 80% train and 20% test
# -------------------------------------
train_size = int(0.8 * len(df_shuffled))
X_train = X_tfidf_shuffled[:train_size]
y_train = y_shuffled[:train_size]
X_test = X_tfidf_shuffled[train_size:]
y_test = y_shuffled[train_size:]

print("\nTraining set size:", X_train.shape, y_train.shape)
print("Testing set size:", X_test.shape, y_test.shape)

# -------------------------------------
# Step 8: Hyperparameter Tuning for Naive Bayes
# -------------------------------------
best_alpha = None
best_score = 0
for alpha in [0.01, 0.1, 0.5, 1.0, 2.0]:
    clf_temp = MultinomialNB(alpha=alpha)
    cv_scores = cross_val_score(clf_temp, X_train, y_train, cv=3, scoring='accuracy')
    mean_cv = cv_scores.mean()
    if mean_cv > best_score:
        best_score = mean_cv
        best_alpha = alpha

print("\nBest alpha found:", best_alpha, "with CV score:", best_score)

# Fit final model with best alpha
clf = MultinomialNB(alpha=best_alpha)
clf.fit(X_train, y_train)

# -------------------------------------
# Step 9: Cross-Validate the final chosen model
# -------------------------------------
cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
print("\nCross-validation scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# -------------------------------------
# Step 10: Evaluate on Test Data
# -------------------------------------
y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
test_acc = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:\n", conf_mat)
print("Test Accuracy:", test_acc)

# -------------------------------------
# Step 11: Test with Custom Comments
# -------------------------------------

custom_comments = [
    # Non-spam comments
    "I really enjoyed this film, it was wonderful!",
    "Thank you for the informative update, very helpful.",
    "Looking forward to your new release, keep up the great work!",
    "I really enjoyed reading this article! The insights were very helpful, and I look forward to learning more on this topic.",
    "What a great product, I will recommend it to my friends.",

    # Spam comments
    "Congratulations! You've won a FREE iPhone, click here!",
    "Act now! Send us your bank details to claim your prize.",
    "Limited offer! Buy now and get a 90% discount, just click the link!",
    "Your account is compromised, verify here with your password immediately!",
    "Get rich quick! Earn thousands from home with minimal effort!"
]

X_custom_counts = count_vect.transform(custom_comments)
X_custom_tfidf = tfidf_transformer.transform(X_custom_counts)
custom_preds = clf.predict(X_custom_tfidf)

print("\nCustom Comments Predictions:")
for comment, pred in zip(custom_comments, custom_preds):
    print(f"Comment: {comment}\nPredicted: {'Spam' if pred == 1 else 'Not Spam'}\n")

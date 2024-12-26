from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_files
from pprint import pprint
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
moviedir = r'/Users/ebbasmac/Desktop/TNM108/LAB4/movie_reviews'

# Loading all files
movie = load_files(moviedir, shuffle=True)
print(f"Number of reviews: {len(movie.data)}")
print(f"Target names (classes): {movie.target_names}")

# Split data into training and test sets
docs_train, docs_test, y_train, y_test = train_test_split(
    movie.data, movie.target, test_size=0.5, random_state=12)

# SVM Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()), #Turns text into numbers
    ('tfidf', TfidfTransformer()), #Adjusts the numbers to give more importance to unique words
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-6, random_state=42, max_iter=5, tol=None)), #It learns patterns from the processed text data to classify
])

# Train model, allows the SVM pipeline to learn from the training data and provides a simple accuracy score
text_clf.fit(docs_train, y_train)
predicted = text_clf.predict(docs_test)
print(f"SVM Accuracy: {np.mean(predicted == y_test):.4f}")

# Generates a report with detailed performance metrics for each class (e.g., positive and negative in the movie reviews dataset). Compares the true labels (y_test) with the predicted labels 
print("\nClassification Report:")
print(metrics.classification_report(y_test, predicted, target_names=movie.target_names))

# Print confusion matrix
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted)) 
#True Negatives,  False Positives
#False Negatives, True Positives

# Grid Search for parameter tuning
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)], #Decides whether to use: Unigrams ((1, 1)) → Individual words, like "good." Bigrams ((1, 2)) → Two-word phrases, like "very good."
    'tfidf__use_idf': (False, True), #Decides whether to use IDF (Inverse Document Frequency): False: Just counts how often a word appears, True: Also considers how rare a word is 
    'clf__alpha': (1e-5, 1e-6), #Controls how much the model avoids overfitting
}

# Takes 600 training samples and tests all combinations of hyperparameters defined in parameters
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(docs_train[:600], y_train[:600]) 

# Cross-validation score and best parameters
print('\nBest GridSearchCV Score:', gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
    print(f"{param_name}: {gs_clf.best_params_[param_name]}")

# Test on new reviews
reviews_new = [
    'This movie was excellent', 'Absolute joy ride',
    'Steven Seagal was terrible', 'Steven Seagal shone through.',
    'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
    "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
    'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.'
]

# Predict sentiment for new reviews
pred = gs_clf.predict(reviews_new)

# Print predictions for new reviews
print("\nPredictions for new reviews:")
for review, category in zip(reviews_new, pred):
    print(f"{review} => {movie.target_names[category]}")

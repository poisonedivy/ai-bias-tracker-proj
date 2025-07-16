import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import csv

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def csv_tokenizer(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            response = row['response']
            #print(response)
            tokens = nltk.word_tokenize(response)
            # Filter out tokens i don't want stop words, punctuation, and non-alphabet characters
            filtered_sentence = [w.lower() for w in tokens if not w.lower() in stop_words and w not in string.punctuation and w.isalpha()]
            print(filtered_sentence)


class TFIDFClassifier:
    def __init__(self, classifier_type='logistic_regression', max_features=10000, min_df=2, max_df=0.8):
        """
        Initialize the TF-IDF classifier
        
        Args:
            classifier_type: 'naive_bayes', 'logistic_regression', or 'svm'
            max_features: Maximum number of features to use
            min_df: Minimum document frequency for a term to be included
            max_df: Maximum document frequency for a term to be included
        """
        self.classifier_type = classifier_type
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess_text,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        # Initialize classifier based on type
        if classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB()
        elif classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif classifier_type == 'svm':
            self.classifier = SVC(random_state=42, kernel='linear')
        else:
            raise ValueError("classifier_type must be 'naive_bayes', 'logistic_regression', or 'svm'")
    
    def preprocess_text(self, text):
        """
        Preprocess text: tokenize, remove stopwords, punctuation, and stem
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens and stem
        tokens = [
            self.stemmer.stem(w) for w in tokens 
            if w.lower() not in self.stop_words and w not in string.punctuation and w.isalpha()
        ]
        
        return tokens
    
    def fit(self, X_train, y_train):
        """
        Train the classifier
        """
        print("Vectorizing training data...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        print(f"Training completed. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        """
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_tfidf)
    
    def predict_proba(self, X_test):
        """
        Get prediction probabilities (if supported by classifier)
        """
        X_test_tfidf = self.vectorizer.transform(X_test)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X_test_tfidf)
        else:
            raise AttributeError(f"{self.classifier_type} doesn't support probability predictions")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier and print metrics
        """
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def get_feature_importance(self, top_n=20):
        """
        Get most important features for each class (works best with logistic regression)
        """
        if self.classifier_type != 'logistic_regression':
            print("Feature importance works best with logistic regression")
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.classifier, 'coef_'):
            coef = self.classifier.coef_
            classes = self.classifier.classes_
            
            for i, class_name in enumerate(classes):
                print(f"\nTop {top_n} features for class '{class_name}':")
                if len(coef.shape) > 1:
                    class_coef = coef[i]
                else:
                    class_coef = coef[0]
                
                top_indices = np.argsort(np.abs(class_coef))[-top_n:][::-1]
                for idx in top_indices:
                    print(f"  {feature_names[idx]}: {class_coef[idx]:.4f}")


def load_and_preprocess_data(csv_file, text_column='text', label_column='label'):
    """
    Load data from CSV file and preprocess it
    
    Args:
        csv_file: Path to CSV file
        text_column: Name of column containing text data
        label_column: Name of column containing labels
    
    Returns:
        X: List of text samples
        y: List of labels
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        # Remove rows with missing values
        df = df.dropna(subset=[text_column, label_column])
        
        # Extract text and labels
        X = df[text_column].tolist()
        y = df[label_column].tolist()
        
        print(f"Loaded {len(X)} samples from {csv_file}")
        print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def train_and_evaluate_model(csv_file, text_column='text', label_column='label', 
                           classifier_type='logistic_regression', test_size=0.2, random_state=42):
    """
    Complete training and evaluation pipeline
    
    Args:
        csv_file: Path to CSV file with labeled data
        text_column: Name of column containing text data
        label_column: Name of column containing labels
        classifier_type: Type of classifier to use
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Trained classifier instance
    """
    
    # Load data
    X, y = load_and_preprocess_data(csv_file, text_column, label_column)
    
    if X is None or y is None:
        return None
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Initialize and train classifier
    classifier = TFIDFClassifier(classifier_type=classifier_type)
    classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    accuracy = classifier.evaluate(X_test, y_test)
    
    # Show feature importance (if using logistic regression)
    if classifier_type == 'logistic_regression':
        classifier.get_feature_importance()
    
    return classifier


def predict_single_text(classifier, text):
    """
    Predict bias for a single text sample
    
    Args:
        classifier: Trained TFIDFClassifier instance
        text: Text to classify
    
    Returns:
        prediction: Predicted class
        probabilities: Class probabilities (if available)
    """
    prediction = classifier.predict([text])[0]
    
    try:
        probabilities = classifier.predict_proba([text])[0]
        prob_dict = dict(zip(classifier.classifier.classes_, probabilities))
        return prediction, prob_dict
    except:
        return prediction, None








def main():
    classifier = train_and_evaluate_model(
    csv_file='C:/Users/jtist/Desktop/work/all labeled responses - Sheet1.csv',
    text_column='response',  # Adjust to your column name
    label_column='economic human label',  # Adjust to your column name
    classifier_type='logistic_regression'
)

main()
import pandas as pd  
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():

    print("Loading dataset...")

    df = pd.read_excel("drugsCom_raw.xlsx")

    df.drop_duplicates(inplace=True)
    df['condition'] = df['condition'].fillna("Unknown")

    conditions = ["Depression", "High Blood Pressure", "Diabetes, Type 2"]
    df = df[df['condition'].isin(conditions)]

    label_encoder = LabelEncoder()
    df['condition_encoded'] = label_encoder.fit_transform(df['condition'])

    X = df['review']
    y = df['condition_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipelines = {
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                token_pattern=r'[a-zA-Z]+',
                max_df=0.9,
                min_df=5,
                ngram_range=(1,2)
            )),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        "Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                token_pattern=r'[a-zA-Z]+',
                max_df=0.9,
                min_df=5,
                ngram_range=(1,2)
            )),
            ('classifier', MultinomialNB())
        ]),
        "Linear SVM": Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                token_pattern=r'[a-zA-Z]+',
                max_df=0.9,
                min_df=5,
                ngram_range=(1,2)
            )),
            ('classifier', LinearSVC())
        ]),
        "Random Forest": Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                token_pattern=r'[a-zA-Z]+',
                max_df=0.9,
                min_df=5,
                ngram_range=(1,2)
            )),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
    }

    param_grids = {
        "Logistic Regression": {'classifier__C': [0.1, 1, 10]},
        "Naive Bayes": {'classifier__alpha': [0.1, 0.5, 1]},
        "Linear SVM": {'classifier__C': [0.1, 1, 10]},
        "Random Forest": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
    }

    results = []
    best_model = None
    best_accuracy = 0

    print("Training models with GridSearch...")

    for name in pipelines.keys():

        grid = GridSearchCV(
            pipelines[name],
            param_grids[name],
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = grid.best_estimator_

        results.append({
            "Model": name,
            "Best CV Score": grid.best_score_,
            "Test Accuracy": accuracy,
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        })

    comparison_df = pd.DataFrame(results).sort_values(
        by="Test Accuracy", ascending=False
    )

    print(comparison_df)

    model_package = {
        "model": best_model,
        "label_encoder": label_encoder
    }

    joblib.dump(model_package, "drug_condition_pipeline.pkl")

    print("Model saved successfully.")


if __name__ == "__main__":
    main()
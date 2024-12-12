#Created by Matthew Jenkins 0n 12/8/24

#
# Model 1: Linear Regression, I chose this model because I needed a baseline model to determine the quality of the simple linear relationships for my dataset.
#

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SpyPlaneLinearRegression:
    def __init__(self, features_file, train_file):
        self.features_file = features_file
        self.train_file = train_file
        self.model = None

    def load_and_prepare_data(self):
        features = pd.read_csv(self.features_file)
        train = pd.read_csv(self.train_file)
        
        labeled_data = pd.merge(train, features, on='adshex')

        labeled_data['class'] = (labeled_data['class'] == 'surveil').astype(int)
        
        self.X_labeled = labeled_data.drop(columns=['adshex', 'class', 'type'])
        self.y_labeled = labeled_data['class']
        
        self.X_full = features.drop(columns=['adshex', 'type'], errors='ignore') 
        self.features = features

    def train_and_evaluate_model(self):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X_labeled, self.y_labeled, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_train_pred = (self.model.predict(X_train) > 0.5).astype(int)
        y_val_pred = (self.model.predict(X_val) > 0.5).astype(int)
        y_test_pred = (self.model.predict(X_test) > 0.5).astype(int)

        self.train_accuracy = accuracy_score(y_train, y_train_pred)
        self.val_accuracy = accuracy_score(y_val, y_val_pred)
        self.test_accuracy = accuracy_score(y_test, y_test_pred)

    def predict_full_dataset(self):
        self.predictions = (self.model.predict(self.X_full) > 0.5).astype(int)

        ground_truth = pd.read_csv(self.train_file)
        ground_truth['class'] = (ground_truth['class'] == 'surveil').astype(int)

        prediction_results = pd.merge(self.features, ground_truth[['adshex', 'class']], on='adshex', how='left')
        prediction_results['predicted'] = self.predictions

        prediction_results.to_csv("model_1_results.csv", index=False)
        
        return prediction_results

    def save_accuracies(self):
        with open("model_1_accuracies.txt", "w") as file:
            file.write("Model Accuracies:\n")
            file.write(f"Training Accuracy: {self.train_accuracy:.2%}\n")
            file.write(f"Validation Accuracy: {self.val_accuracy:.2%}\n")
            file.write(f"Testing Accuracy: {self.test_accuracy:.2%}\n")

    def display_results(self):
        print("Model Accuracies:")
        print(f"Training Accuracy: {self.train_accuracy:.2%}")
        print(f"Validation Accuracy: {self.val_accuracy:.2%}")
        print(f"Testing Accuracy: {self.test_accuracy:.2%}")
  
        self.save_accuracies()
        self.predict_full_dataset()
        print("\nResults saved to 'model_1_results.csv' and 'model_1_accuracies.txt'.")

if __name__ == "__main__":
    features_file = "planes_features.csv"
    train_file = "train.csv"
    
    spy_plane_lr = SpyPlaneLinearRegression(features_file, train_file)
    spy_plane_lr.load_and_prepare_data()
    spy_plane_lr.train_and_evaluate_model()
    spy_plane_lr.display_results()

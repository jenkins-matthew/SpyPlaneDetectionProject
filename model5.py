#Created by Matthew Jenkins on 12/8/24

#
# Model 5: Decision Tree, I chose this model because I wanted to be able to create a decision based process for determining how to detect planes based on the dataset
#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class SpyPlaneDecisionTree:
    def __init__(self, features_file, train_file):
        self.features_file = features_file
        self.train_file = train_file
        self.model = None

    def load_and_prepare_data(self):
        features = pd.read_csv(self.features_file)
        train = pd.read_csv(self.train_file)
        data = pd.merge(train, features, on="adshex")
        data["class"] = (data["class"] == "surveil").astype(int)
        
        X = data.drop(columns=["adshex", "class", "type"])
        y = data["class"]
        
        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=42)
        
        self.X_full = features.drop(columns=["adshex", "type"], errors="ignore")
        self.features = features

    def train_and_evaluate_model(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        train_predictions = self.model.predict(self.X_train)
        val_predictions = self.model.predict(self.X_val)
        test_predictions = self.model.predict(self.X_test)
        
        self.train_accuracy = accuracy_score(self.y_train, train_predictions)
        self.val_accuracy = accuracy_score(self.y_val, val_predictions)
        self.test_accuracy = accuracy_score(self.y_test, test_predictions)

        test_results = pd.DataFrame(self.X_test, columns=self.X_train.columns)
        test_results["Ground Truth"] = self.y_test.values
        test_results["Predictions"] = test_predictions
        test_results.to_csv("model_5_results.csv", index=False)

    def predict_full_dataset(self):
        full_predictions = self.model.predict(self.X_full)
        
        ground_truth = pd.read_csv(self.train_file)
        ground_truth["class"] = (ground_truth["class"] == "surveil").astype(int)

        prediction_results = self.features.copy()
        prediction_results["class"] = prediction_results["adshex"].map(ground_truth.set_index("adshex")["class"])
        prediction_results["predicted"] = full_predictions

        prediction_results.to_csv("model_5_results.csv", index=False)
        
        total_planes = len(self.features)
        detected_planes = sum(full_predictions)
        detected_percentage = (detected_planes / total_planes) * 100
        true_positives = prediction_results[(prediction_results["predicted"] == 1) & (prediction_results["class"] == 1)].shape[0]

        return total_planes, detected_planes, detected_percentage, true_positives

    def save_accuracies(self):
        with open("model_5_accuracies.txt", "w") as file:
            file.write("Model Accuracies:\n")
            file.write(f"Training Accuracy: {self.train_accuracy:.2%}\n")
            file.write(f"Validation Accuracy: {self.val_accuracy:.2%}\n")
            file.write(f"Testing Accuracy: {self.test_accuracy:.2%}\n")

    def display_results(self):
        print("Model Accuracies:")
        print(f"Training Accuracy: {self.train_accuracy * 100:.2f}%")
        print(f"Validation Accuracy: {self.val_accuracy * 100:.2f}%")
        print(f"Testing Accuracy: {self.test_accuracy * 100:.2f}%")
        
        self.save_accuracies()
        
        total_planes, detected_planes, detected_percentage, true_positives = self.predict_full_dataset()
        print("\nPlane Detection Statistics for Full Dataset:")
        print(f"Total Number of Planes: {total_planes}")
        print(f"Number of Planes Detected: {detected_planes}")
        print(f"Percentage of Planes Detected: {detected_percentage:.2f}%")
        print(f"Number of Detected Planes That Are Actual Spy Planes: {true_positives}")
        print("\nResults saved to 'model_5_results.csv' and 'model_5_accuracies.txt'.")

        plt.figure(figsize=(20, 10))
        plot_tree(self.model, filled=True, feature_names=self.X_train.columns, class_names=["Not Surveil", "Surveil"])
        plt.title("Decision Tree Visualization")
        plt.show()

if __name__ == "__main__":
    features_file = "planes_features.csv"
    train_file = "train.csv"

    spy_tree = SpyPlaneDecisionTree(features_file, train_file)
    spy_tree.load_and_prepare_data()
    spy_tree.train_and_evaluate_model()
    spy_tree.display_results()

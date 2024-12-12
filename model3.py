#Created by Matthew Jenkins on 12/8/24

#
# Model 3: XGBoost, I chose this model for its high performance and also because it handles imbalanced data the best
#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

class SpyPlaneXGBoost:
    def __init__(self, features_file, train_file):
        self.features_file = features_file
        self.train_file = train_file
        self.model = None

    def load_and_prepare_data(self):
        features = pd.read_csv(self.features_file)
        train = pd.read_csv(self.train_file)
 
        labeled_data = pd.merge(train, features, on="adshex")
        
        labeled_data["class"] = (labeled_data["class"] == "surveil").astype(int)
        
        self.X_labeled = labeled_data.drop(columns=["adshex", "class", "type"])
        self.y_labeled = labeled_data["class"]
        
        self.X_full = features.drop(columns=["adshex", "type"], errors="ignore")
        self.features = features

    def train_and_evaluate_model(self):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X_labeled, self.y_labeled, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "eta": 0.1,
            "seed": 42
        }

        evallist = [(dtrain, "train"), (dval, "validation")]
        self.model = xgb.train(params, dtrain, num_boost_round=100, evals=evallist, early_stopping_rounds=10, verbose_eval=False)
        
        train_predictions = (self.model.predict(dtrain) > 0.5).astype(int)
        val_predictions = (self.model.predict(dval) > 0.5).astype(int)
        test_predictions = (self.model.predict(dtest) > 0.5).astype(int)
        
        self.train_accuracy = accuracy_score(y_train, train_predictions)
        self.val_accuracy = accuracy_score(y_val, val_predictions)
        self.test_accuracy = accuracy_score(y_test, test_predictions)

        test_results = pd.DataFrame(X_test, columns=self.X_labeled.columns)
        test_results['Ground Truth'] = y_test.values
        test_results['Predictions'] = test_predictions
        test_results.to_csv("model_3_results.csv", index=False)

    def predict_full_dataset(self):
        dfull = xgb.DMatrix(self.X_full)
        full_predictions = (self.model.predict(dfull) > 0.5).astype(int)
        
        ground_truth = pd.read_csv(self.train_file)
        ground_truth["class"] = (ground_truth["class"] == "surveil").astype(int)

        prediction_results = self.features.copy()
        prediction_results["class"] = prediction_results["adshex"].map(ground_truth.set_index("adshex")["class"])
        prediction_results["predicted"] = full_predictions
        
        prediction_results.to_csv("model_3_results.csv", index=False)
        
        total_planes = len(self.features)
        detected_planes = np.sum(full_predictions)
        detected_percentage = (detected_planes / total_planes) * 100
        true_positives = prediction_results[(prediction_results["predicted"] == 1) & (prediction_results["class"] == 1)].shape[0]

        return total_planes, detected_planes, detected_percentage, true_positives

    def save_accuracies(self):
        with open("model_3_accuracies.txt", "w") as file:
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
        
        total_planes, detected_planes, detected_percentage, true_positives = self.predict_full_dataset()
        print("\nPlane Detection Statistics for Full Dataset:")
        print(f"Total Number of Planes: {total_planes}")
        print(f"Number of Planes Detected: {detected_planes}")
        print(f"Percentage of Planes Detected: {detected_percentage:.2f}%")
        print(f"Number of Detected Planes That Are Actual Spy Planes: {true_positives}")
        print("\nResults saved to 'model_3_results.csv' and 'model_3_accuracies.txt'.")

if __name__ == "__main__":
    features_file = "planes_features.csv"
    train_file = "train.csv"

    spy_plane_xgb = SpyPlaneXGBoost(features_file, train_file)
    spy_plane_xgb.load_and_prepare_data()
    spy_plane_xgb.train_and_evaluate_model()
    spy_plane_xgb.display_results()

#Created by Matthew Jenkins 0n 12/8/24

#
# Model 2: Multi-Layer Neural Network, I chose this model because of its its ability to capture complex, non-linear patterns within my dataset.
#

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SpyPlaneMultiLayerNN:
    def __init__(self, features_file, train_file, model_path="saved_model.pth", learning_rate=0.01, num_epochs=100):
        self.features_file = features_file
        self.train_file = train_file
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.val_accuracy = None
        self.train_accuracies = None
        self.val_accuracies = None
        self.test_accuracies = None
        self.model_2_accuracies = {}
        self.model_2_results = {}

    def load_and_prepare_data(self):
        features = pd.read_csv(self.features_file)
        train = pd.read_csv(self.train_file)

        labeled_data = pd.merge(train, features, on="adshex")
        labeled_data["class"] = (labeled_data["class"] == "surveil").astype(int)
        
        self.X_labeled = labeled_data.drop(columns=["adshex", "class", "type"])
        self.y_labeled = labeled_data["class"]
        
        self.X_full = features.drop(columns=["adshex", "type"], errors="ignore")
        self.features = features

        X_train, X_temp, y_train, y_temp = train_test_split(self.X_labeled, self.y_labeled, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        self.X_train_T = torch.FloatTensor(X_train.values)
        self.y_train_T = torch.LongTensor(y_train.values)
        self.X_val_T = torch.FloatTensor(X_val.values)
        self.y_val_T = torch.LongTensor(y_val.values)
        self.X_test_T = torch.FloatTensor(X_test.values)
        self.y_test_T = torch.LongTensor(y_test.values)

    def train_and_evaluate_model(self):
        input_dim = self.X_train_T.shape[1]
        output_dim = 2
        self.model = self.MultiLayerNet(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        train_losses = np.zeros(self.num_epochs)
        val_losses = np.zeros(self.num_epochs)
        test_losses = np.zeros(self.num_epochs)
        self.train_accuracies = np.zeros(self.num_epochs)
        self.val_accuracies = np.zeros(self.num_epochs)
        self.test_accuracies = np.zeros(self.num_epochs)

        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train_T)
            loss = self.criterion(outputs, self.y_train_T)
            loss.backward()
            self.optimizer.step()
            train_losses[epoch] = loss.item()
            _, train_pred = torch.max(outputs, 1)
            self.train_accuracies[epoch] = (train_pred == self.y_train_T).sum().item() / self.y_train_T.size(0)

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val_T)
                val_loss = self.criterion(val_outputs, self.y_val_T)
                val_losses[epoch] = val_loss.item()
                _, val_pred = torch.max(val_outputs, 1)
                self.val_accuracies[epoch] = (val_pred == self.y_val_T).sum().item() / self.y_val_T.size(0)

            with torch.no_grad():
                test_outputs = self.model(self.X_test_T)
                test_loss = self.criterion(test_outputs, self.y_test_T)
                test_losses[epoch] = test_loss.item()
                _, test_pred = torch.max(test_outputs, 1)
                self.test_accuracies[epoch] = (test_pred == self.y_test_T).sum().item() / self.y_test_T.size(0)

        self.train_accuracy = self.train_accuracies[-1]
        self.val_accuracy = self.val_accuracies[-1]
        self.test_accuracy = self.test_accuracies[-1]

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'test_accuracies': self.test_accuracies,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'test_accuracy': self.test_accuracy
        }, self.model_path)
        print(f"Model and accuracies saved to {self.model_path}")

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.train_accuracies, label="Train Accuracy")
        plt.plot(self.val_accuracies, label="Validation Accuracy")
        plt.plot(self.test_accuracies, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Epochs")
        plt.legend()
        plt.show()

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            input_dim = self.X_train_T.shape[1]
            output_dim = 2
            self.model = self.MultiLayerNet(input_dim, output_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.train_accuracies = checkpoint.get('train_accuracies', None)
            self.val_accuracies = checkpoint.get('val_accuracies', None)
            self.test_accuracies = checkpoint.get('test_accuracies', None)
            self.train_accuracy = checkpoint.get('train_accuracy', None)
            self.val_accuracy = checkpoint.get('val_accuracy', None)
            self.test_accuracy = checkpoint.get('test_accuracy', None)

            self.model.eval()
            print(f"Model loaded from {self.model_path}")

            if self.train_accuracy is None or self.val_accuracy is None or self.test_accuracy is None:
                print("Warning: Accuracies not found in checkpoint. Retrain the model to update these values.")
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def predict_full_dataset(self):
        X_full_T = torch.FloatTensor(self.X_full.values)
        with torch.no_grad():
            full_predictions = self.model(X_full_T)
            _, full_pred = torch.max(full_predictions, 1)
        
        total_planes = len(self.features)
        detected_planes = full_pred.sum().item()
        detected_percentage = (detected_planes / total_planes) * 100

        ground_truth = pd.read_csv(self.train_file)
        ground_truth["class"] = (ground_truth["class"] == "surveil").astype(int)

        prediction_results = pd.merge(self.features, ground_truth[["adshex", "class"]], on="adshex", how="left")
        prediction_results["predicted"] = full_pred.numpy()

        prediction_results.to_csv("model_2_results.csv", index=False)

        true_positives = prediction_results[(prediction_results["predicted"] == 1) & (prediction_results["class"] == 1)].shape[0]

        print("\nPlane Detection Statistics for Full Dataset:")
        print(f"Total Number of Planes: {total_planes}")
        print(f"Number of Planes Detected as Possible Spy Planes: {detected_planes}")
        print(f"Percentage of Planes Detected: {detected_percentage:.2f}%")
        print(f"Number of Detected Planes That Are Actual Spy Planes: {true_positives}")

        self.model_2_results = {
            "Total Planes": total_planes,
            "Detected Planes": detected_planes,
            "Detection Percentage": detected_percentage,
            "True Positives": true_positives
        }
        return self.model_2_results

    def display_results(self):
        print("Model Accuracies:")
        if self.train_accuracy is not None and self.val_accuracy is not None and self.test_accuracy is not None:
            print(f"Training Accuracy: {self.train_accuracy:.2%}")
            print(f"Validation Accuracy: {self.val_accuracy:.2%}")
            print(f"Testing Accuracy: {self.test_accuracy:.2%}")
            
            self.model_2_accuracies = {
                "Training Accuracy": self.train_accuracy,
                "Validation Accuracy": self.val_accuracy,
                "Testing Accuracy": self.test_accuracy
            }

            with open("model_2_accuracies.txt", "w") as f:
                f.write("Model Accuracies:\n")
                for key, value in self.model_2_accuracies.items():
                    f.write(f"{key}: {value:.2%}\n")
        else:
            print("Accuracies are not available.")

        self.predict_full_dataset()

    class MultiLayerNet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

if __name__ == "__main__":
    features_file = "planes_features.csv"
    train_file = "train.csv"
    
    spy_plane_nn = SpyPlaneMultiLayerNN(features_file, train_file)
    spy_plane_nn.load_and_prepare_data()
    if input("Load saved model? (y/n): ").strip().lower() == 'y':
        spy_plane_nn.load_model()
    else:
        spy_plane_nn.train_and_evaluate_model()
    spy_plane_nn.display_results()

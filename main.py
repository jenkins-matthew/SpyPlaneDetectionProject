#Created by Matthew Jenkins on 12/8/24

#
# USEREnd menu for running models and additional functionality, showing accuracy plot and dataset results
#

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def menu():
    while True:
        print("\n--- Spy Plane Detection Menu ---")
        print("1. Run Model 1 (Linear Regression)")
        print("2. Run Model 2 (Multi-Layer Neural Network)")
        print("3. Run Model 3 (XGBoost)")
        print("4. Run Model 4 (Logistic Regression)")
        print("5. Run Model 5 (Decision Tree)")
        print("6. Plot Accuracy vs Model")
        print("7. Print Detection Results for Each Model")
        print("8. Exit")

        choice = input("Select an option: ").strip()
        
        if choice == "1":
            os.system("python model1.py")
        elif choice == "2":
            os.system("python model2.py")
        elif choice == "3":
            os.system("python model3.py")
        elif choice == "4":
            os.system("python model4.py")
        elif choice == "5":
            os.system("python model5.py")
        elif choice == "6":
            plot_accuracies()
        elif choice == "7":
            print_detection_results()
        elif choice == "8":
            print("Exiting the menu.")
            break
        else:
            print("Invalid choice. Please try again.")

def plot_accuracies():
    accuracy_files = [
        "model_1_accuracies.txt",
        "model_2_accuracies.txt",
        "model_3_accuracies.txt",
        "model_4_accuracies.txt",
        "model_5_accuracies.txt",
    ]
    models = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]
    training_acc = []
    validation_acc = []
    testing_acc = []

    for file in accuracy_files:
        if os.path.exists(file):
            try:
                with open(file, "r") as f:
                    lines = f.readlines()
                    if len(lines) < 4:
                        raise ValueError(f"File {file} does not have the expected number of lines.")
                    training_acc.append(float(lines[1].split(":")[1].strip().strip("%")) / 100)
                    validation_acc.append(float(lines[2].split(":")[1].strip().strip("%")) / 100)
                    testing_acc.append(float(lines[3].split(":")[1].strip().strip("%")) / 100)
            except (IndexError, ValueError) as e:
                print(f"Error processing file {file}: {e}")
                return
        else:
            print(f"File {file} not found. Skipping.")
            return

    plt.figure(figsize=(10, 6))
    plt.plot(models, training_acc, marker="o", label="Training Accuracy")
    plt.plot(models, validation_acc, marker="o", label="Validation Accuracy")
    plt.plot(models, testing_acc, marker="o", label="Testing Accuracy")
    plt.title("Accuracy vs Model")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

def print_detection_results():
    result_files = [
        "model_1_results.csv",
        "model_2_results.csv",
        "model_3_results.csv",
        "model_4_results.csv",
        "model_5_results.csv",
    ]
    models = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]

    try:
        feds = pd.read_csv("feds.csv")
    except FileNotFoundError:
        print("feds.csv not found. Please ensure the file is in the same directory as this script.")
        return

    for model, file in zip(models, result_files):
        if os.path.exists(file):
            results = pd.read_csv(file)
            total_planes = results.shape[0]
            detected_planes = results["predicted"].sum()
            spy_planes = results[results["adshex"].isin(feds["adshex"])]["predicted"].sum()

            print(f"\n{model} Detection Results:")
            print(f"Total Number of Planes: {total_planes}")
            print(f"Number of Planes Detected as Possible Spy Planes: {detected_planes}")
            print(f"Number of Planes That Are Actual Spy Planes: {spy_planes}")
        else:
            print(f"File {file} not found. Skipping.")

if __name__ == "__main__":
    menu()

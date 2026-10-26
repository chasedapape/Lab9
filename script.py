# Eric Reid
# October 24, 2026
# HSC 4933 - Python for Health Analytics
# Lab 9 - Logistic and Count-Variable Regression
# Purpose is to use a dataset to run regression and save the results

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Ask user which dataset they would like to use
dataset = input("Enter dataset filename: ")

# Load dataset
df = pd.read_csv(dataset)

# Clean columns w/ snakecase
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Displays the available columns
print("\nAvailable columns:")
print(df.columns.tolist())

# Ask the user to select predictor variables
predictors = input("Enter predictors(separated by comma): ")
predictors = [col.strip() for col in predictors.split(",")] # Converts data to list
response = input("Enter the response column: ").strip()


# Keeps only the columns selected fro regression
selected_columns = predictors + [response]

# Removes rows with missing values
clean_df = df.dropna(subset=selected_columns)

regression_type = input("\nEnter regression type (logistic or poisson): ").strip().lower()
if regression_type == "logistic":
    print("Running Logistic Regression... ")
    if not clean_df[response].isin([0,1]).all():
        print("Error: Invalid response")

    else:
        # Separate predictors(x) and response(y)
        X = clean_df[predictors]
        Y = clean_df[response]

        #Split data into 80% training and 20% testing
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        #Creates and trains the logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)

        # Predicts the outcomes using testing data
        y_pred = model.predict(X_test)
        print("Logistic Regression Completed.")

        # Calculates model accuracy
        accuracy = accuracy_score(Y_test, y_pred)

        # Classification report
        report = classification_report(Y_test, y_pred)

        # Confusion matrix
        matrix = confusion_matrix(Y_test, y_pred)

        #Add intercept for statistical model
        X_constant = sm.add_constant(X)

        # Fit logistic regression using statdels
        stats_model = sm.Logit(Y, X_constant).fit(disp=False)

        # Calculate pseudo R-squared
        pseudo_r2 = stats_model.prsquared

# Combine regression results into one output
        results = (
            f"Logistic Regression Results\n"
            f"Predictors: {predictors}\n"
            f"Response: {response}\n\n"
            f"Model Accuracy: {accuracy:.2%}\n\n"
            f"Classification Report:\n{report}\n"
            f"Confusion Matrix:\n{matrix}\n\n"
            f"Pseudo R^2: {pseudo_r2:.4f}\n"
        )

        print(results)
        # Ask if user wants to save results
        save = input("Would you like to save these results? (y/n): ").strip().lower()

        if save == "y":
            filename = input("Enter filename: ").strip()

            if not filename.endswith(".txt"):
                filename += ".txt"

            with open(filename, "w") as file:
                file.write(results)
            print(f"Results saved successfully to {filename}.")

elif regression_type == "poisson":
    print("Running Poisson Regression... ")

    # Separate predictors and response
    X = clean_df[predictors]
    Y = clean_df[response]

    #Add an intercept to the predictor variables
    X_constant = sm.add_constant(X)

    poisson_model = sm.Poisson(Y, X_constant).fit()

    # Display the regression summary
    print(poisson_model.summary())

    # Calculates VIF for predictor variables
    vif = pd.DataFrame()

    vif["Predictor"] = X.columns

    vif["VIF"] = [
        variance_inflation_factor(X_constant.values, i)
        for i in range(1, X_constant.shape[1])
    ]

    # Displays the VIF results
    print("\nVariance Inflation Factors:")
    print(vif)

    # Combine Poisson results for saving
    results = (
        f"Poisson Regression Results\n"
        f"Predictors: {predictors}\n"
        f"Response: {response}\n\n"
        f"{poisson_model.summary()}\n\n"
        f"Variance Inflation Factors:\n{vif.to_string(index=False)}\n"
    )

    # Ask user whether to save the results
    save = input("\nWould you like to save the results? (y/n): ").strip().lower()

    if save == "y":
        filename = input("Enter filename: ").strip()

        if not filename.endswith(".txt"):
            filename += ".txt"

        with open(filename, "w") as file:
            file.write(results)

        print(f"Results saved successfully to {filename}")
else:
    print("Try again.")
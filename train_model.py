import os
from pathlib import Path
from data_loader import data_loader
from baseline_logistic import train_logistic_regression
from random_forest_model import train_random_forest

def train_and_evaluate():

    X_train, y_train, X_val, y_val, X_test, y_test = data_loader()
    
    # Logistic Regression
    train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    # Random Forest
    train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    train_and_evaluate()
    

    
    

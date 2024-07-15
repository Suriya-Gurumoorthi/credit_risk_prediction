import os
import sys
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from src.exception import CustomException
from sklearn.metrics import precision_score
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Split training and testing data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision tree": DecisionTreeClassifier(),
                "Support Vector Machine": SVC(),
                "Naive Bayes": GaussianNB(),
                "Logistic Regression": LogisticRegression(),
            }

#            params = {
#                'Random Forest': {
#                    'n_estimators': [100, 200, 300],
#                   'max_depth': [None, 5, 10, 15],
#                    'min_samples_split': [2, 5, 10],
#                    'min_samples_leaf': [1, 2, 4],
#                    'max_features': ['sqrt', 'log2', None]
#                },
#                'Decision tree': {
#                    'max_depth': [None, 5, 10, 15, 20],
#                    'min_samples_split': [2, 5, 10],
#                    'min_samples_leaf': [1, 2, 4],
#                    'max_features': ['sqrt', 'log2', None]
#                },
#                'Support Vector Machine': {
#                    'C': [0.1, 1, 10, 100],
#                    'kernel': ['rbf', 'linear', 'poly'],
#                    'gamma': ['scale', 'auto', 0.1, 1],
#                    'degree': [2, 3, 4]  # Only relevant for poly kernel
#                },
#                'Naive Bayes': {
#                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
#                },
#                'Logistic Regression': {
#                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                    'penalty': ['l1', 'l2', 'elasticnet', None],
#                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
#                    'max_iter': [100, 200, 300]
#               }
#            }
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found.")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            precision = precision_score(y_test,predicted)
            return precision

        except Exception as e:
            raise CustomException(e, sys)
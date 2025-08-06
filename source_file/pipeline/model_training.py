import numpy as np
import os
import sys 
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from source_file.exception import CustomException
from source_file.logger import logging
from source_file.utils import save_object,evaluate_models


class ModelTrainerConfig:
    trained_model_file_path = os.path.join ("artifacts","diabetes_diseases.pkl")

class ModelTrainer:
    def __init__(self):
     self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer (self,train_array,test_array):
       try:
          logging.info("Split training and test input data")
          diabetes_col_index = -1  # 3rd last column

          # Feature columns: all except diabetes
          X_train = np.delete(train_array, diabetes_col_index, axis=1)
          y_train = train_array[:, diabetes_col_index]

          X_test = np.delete(test_array, diabetes_col_index, axis=1)
          y_test = test_array[:, diabetes_col_index]
          
          models = {
             "Random Forest" : RandomForestClassifier(),
             "Decision Tree" : DecisionTreeClassifier(),
             #"Gradient Boosting" : GradientBoostingClassifier(),
             "Logistic Regression" : LogisticRegression(),
             "K-NeighborsClassifier" : KNeighborsClassifier(),
             "XGBClassifier" : XGBClassifier(),
             "CatBoosting Classifier" : CatBoostClassifier(silent = True,class_weights='Balanced'),
             #"Adaboost Classifier" : AdaBoostClassifier(),
             "Support vector machine" : SVC()
               
            }
          params= {
                "Decision Tree": {
                     "max_depth": [5,10],
                     
                     "criterion": ['gini'],
                     "class_weight": ['balanced']
                },
                "Random Forest":{
                      "n_estimators": [100, 200],
                       "max_depth": [10,20],
                      "min_samples_split": [2],
                      "min_samples_leaf": [1],
                      "criterion": ['gini']
                  },
                
               
                "Gradient Boosting":{
                     
                    'learning_rate':[.1,.01],
                    'subsample':[0.8,1],
                     'criterion':['entropy'],
                     'max_features':['sqrt'],
                    'n_estimators': [100,200]
                },
                "K-NeighborsClassifier":{
                    "n_neighbors": [5,7],
                    "weights": ["uniform", "distance"],
                    
                    "p": [1, 2]
                    
                },

                  "Logistic Regression":{
                         "penalty": ["l1", "l2",],
                         "C": [0.1, 1, 10],
                         "solver": ["liblinear"],
                         
                        "class_weight": ["balanced"]
                  },
                
                "XGBClassifier":{
                   "learning_rate": [0.01,0.1],
                    "max_depth": [3, 5],
                     "n_estimators": [100],
                     "subsample": [0.8],
                    "colsample_bytree": [0.7]
                  },
                "CatBoosting Classifier":{
                    "depth": [4, 6],
                    "learning_rate": [0.03],
                    "iterations": [300],
                    "l2_leaf_reg": [3]
                  },
                #"AdaBoost Classifier":{
                 #   'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                 #   'n_estimators': [8,16,32,64,128,256,512]
                #},
                  "Support Vector Machine":{
                    "kernel": ["rbf", "linear"],
                    "C": [0.1, 1, 10],
                    "class_weight": ["balanced"]

                   
                   }
                
                
         }
          
          model_report : dict  = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test = y_test,models = models,params = params)
          best_model_score = max(model_report.values())

          best_model_name = list(model_report.keys())[
             list(model_report.values()).index(best_model_score)
          ]

          best_model = models[best_model_name]

          if best_model_score < 0.6:
             raise CustomException("No best model is found ")
          logging .info(f"Best found model on both training and testing dataset")

          save_object (
             file_path = self.model_trainer_config.trained_model_file_path,
             obj = best_model
            
          ) 
          logging.info("Predicting the test data and the f1 score")
          predicted = best_model.predict(X_test)
          r2_square = f1_score(y_test,predicted,average='binary')

          return r2_square 
              
       except Exception as e:
          raise CustomException(e,sys)

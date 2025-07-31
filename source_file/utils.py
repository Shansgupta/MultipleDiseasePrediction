## Contain the common functionality from which the entire project can use 


import os
import sys
import numpy as np
from source_file.exception import CustomException
import pandas as pd
import dill  ## actually help to create the pickle file 
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import StratifiedKFold
# source_file/utils.py
import pickle
import os
from source_file.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        print(f"✅ Saved object to {file_path}")
    except Exception as e:
        print("❌ Error while saving object:", e)
        raise e

    
    logging.info("Evaluating the model")
def evaluate_models(X_train,y_train,X_test,y_test,models,params):

    try :
      cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      report = {} 

      for i in range(len(list(models))):
        model = list(models.values())[i]

        ## model.fit(X_train, y_train) # Train model
        para=params[list(models.keys())[i]]
        rs = GridSearchCV(model,param_grid=para, 
                        scoring='f1',
                        cv=cv, n_jobs=-1, verbose=2)
        rs.fit(X_train,y_train)
        best_params  = rs.best_params_
        
        model.set_params(** best_params)  ## applies best parameter to the original model
        model.fit(X_train,y_train)

                  # Make predictions
        y_train_pred = model.predict(X_train)
      
        y_test_pred = model.predict(X_test)
    
                   # Evaluate Train and Test dataset
        train_model_score  = f1_score(y_train, y_train_pred)

        test_model_score =  f1_score(y_test, y_test_pred)
        
        report[list(models.keys())[i]] = test_model_score

        return report 
    
    except Exception as e :
      raise CustomException(e,sys)
     
def load_object(file_path):
   try:
      with open (file_path,"rb") as file_obj:
        return dill.load(file_obj)
       
   except Exception as e :
      raise CustomException(e,sys)     
   
   
   ## print(list(models.keys())[i])
    ## model_list.append(list(models.keys())[i])    
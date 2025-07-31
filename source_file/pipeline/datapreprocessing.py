import sys
import os  
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer ## for the missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from imblearn.over_sampling import SMOTE

from source_file.exception import CustomException
from source_file.logger import logging
from source_file.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()

   ## Below fuction is created just create all pickle files which is responsible for converting 
   #   categoricla to numerical while performing the One Hot encoder and Standard scaling
      
    def get_data_transformer_object(self) :

        try:
            numerical_Columns = ["gender","age","hypertension","heart_disease","bmi","HbA1c_level","blood_glucose_level","glucose_hba1c"]

            categorical_columns = [
              "age_group",
              "smoking_history",
            
             ]
            num_pipeline = Pipeline(
                steps=[
                    
                    ("scaler",StandardScaler())
                ]

            )
            cat_pipeline = Pipeline(
                steps=[
                    
                     ("onehotencoder",OneHotEncoder()),
                     ("scaler",StandardScaler(with_mean = False))

                ]
            )
              
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_Columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_tranformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and the test data cpmpletely")
            logging.info("Obtaining the preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'diabetes'
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info( f"Applying preprocessing object on training dataframe and testing dataframe " )

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)
            logging.info("Doing SMOTE operation in traindata")
            smote = SMOTE(sampling_strategy=0.5, random_state=42)
            input_feature_train_arr, target_feature_train_df = smote.fit_resample(
            input_feature_train_arr, target_feature_train_df
            ) 
            logging.info("Combining both indepenedent and dependent features on both training and testing")
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            print(type(train_arr))
            logging.info(f"Saved preprocessing object")
        
            save_object(
             file_path = self.data_transformation_config.preprocessor_obj_file_path,
             obj = preprocessing_obj

            )

            return( 
            train_arr,
            test_arr,
            )
           # print("📦 Preparing to save preprocessor object...")
            #print("🔧 Path:", self.data_transformation_config.preprocessor_obj_file_path)

        

        except Exception as e :
          
           raise CustomException(e,sys)

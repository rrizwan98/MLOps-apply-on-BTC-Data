import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

logging.info("necessary libraries are installed in data_ingestion.py")

# split the data and save it to artifacts folder
@dataclass
class DataIngestionConfig:
    X_train_data_path: str = os.path.join('artifacts','X_train.npy')
    y_train_data_path: str = os.path.join('artifacts','y_train.npy') 
    X_test_data_path: str = os.path.join('artifacts','X_test.npy')
    y_test_data_path: str = os.path.join('artifacts','y_test.npy') 
    row_data_path: str = os.path.join('artifacts','main_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\dataset\main.csv')
            logging.info('Read the data as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.X_train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.row_data_path,index=False, header=True)

#prepare train images
            logging.info('Start Image pre processing')
            X = []
            for i in tqdm(range(df.shape[0])):
                img = tf.keras.utils.load_img("notebook/" + df['img_paths'][i],target_size=(400,400,3))
                img = tf.keras.utils.img_to_array(img)
                img = img/255
                X.append(img)
            X= np.array(X)
            print(X.shape)
            

#prepare train labels
            logging.info('Drop column in df to preper labels')
            y = df.drop(['image', 'name', 'author', 'format', 'book_depository_stars', 'price', 'currency', 'old_price', 'isbn', 'img_paths'],axis=1)

# encode class values as integers  
            logging.info('Apply Label encoder and convert yCol into numpy array ')          
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)       
            # y = y.to_numpy()
            print("encoded_labels:",y)
            decoded_labels = encoder.inverse_transform(y)
            print("decoded_labels: ",decoded_labels)

            logging.info("Train test split initited")
            # trian_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

            print(X_train.shape)
            print(y_train.shape)
                
            print(X_test.shape)
            print(y_test.shape)

# Save the data in numpy formate
            logging.info("start saving the data in numpy formate")
            np.save(self.ingestion_config.X_train_data_path,X_train)
            np.save(self.ingestion_config.y_train_data_path,y_train)
            np.save(self.ingestion_config.X_test_data_path,X_test)
            np.save(self.ingestion_config.y_test_data_path,y_test)

            logging.info("Ingestion of the data iss completed")

            return(

                self.ingestion_config.X_train_data_path,
                self.ingestion_config.y_train_data_path,
                self.ingestion_config.X_test_data_path,
                self.ingestion_config.y_test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    obj=DataIngestion()
    X_train_data_path,y_train_data_path,X_test_data_path,y_test_data_path = obj.initiate_data_ingestion()

    # data_transformation=DataTransformation()
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train_data_path,y_train_data_path,X_test_data_path,y_test_data_path))
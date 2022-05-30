


import numpy as np # for numerical operations
import pandas as pd # for handling input data
import matplotlib.pyplot as plt # for data visualization 
import seaborn as sns # for data visualization 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



class preprocess : 
    def __init__ (self,df):
        self.df=df
        
        
        
        
    def cleaning(self):
        self.df["diagnosis"] = self.df["diagnosis"].map({"B": 0, "M": 1})
    ##return true if there is null value
        missing = self.df.isnull().any(axis=1)
        no_of_rows_with_nan = missing.sum()
     ##print(no_of_rows_with_nan)
    
        
        

        
    def split_fun(self):
        df_input =  self.df.drop(columns=['diagnosis','Index'])
        df_output =  self.df['diagnosis']
        self.x_train , self.x_test , self.y_train , self.y_test = train_test_split(df_input,df_output, test_size=0.25, random_state=0)
        print(end='\n')
        print(self.x_train.shape)
        print(self.y_train.shape)
        print("-----------------")
        print(self.x_test.shape)
        print(self.y_test.shape)
        print(end='\n')
    
        
  
    
       
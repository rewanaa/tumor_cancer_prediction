import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
from preprocessing import preprocess
class support_vector_machine:
    model2_filename = "svm_model_job_lib.job_lib" 
    result= []
    y_pred_svm_test=[]
    model2 = svm.SVC(kernel='linear', random_state=0)
    def __init__ (self ,x_train , x_test , y_train ,  y_test):
        self.x_train =x_train 
        self.x_test = x_test
        self. y_train= y_train 
        self.y_test=y_test    
    def classification_svm (self):
    #Train data
        self.model2.fit(self.x_train, self.y_train )
    # Predicting the Training data
        y_pred_svm_train = self.model2.predict(self.x_train)      
    # Predicting the Test data
        self.y_pred_svm_test = self.model2.predict(self.x_test)
        
        # Making the Confusion Matrix for SVM
        Confusion_Matrix_svm = confusion_matrix(self.y_test, self.y_pred_svm_test)
       #Model Accuracy for SVM
        svm_accuracy = metrics.accuracy_score(self.y_test, self.y_pred_svm_test)     
        # save model 
        joblib.dump( self.model2, self.model2_filename)
        print('Model svm is saved into to disk successfully')
        print( end = '\n')
            
        print ("SVM MODEL :: ")
        print( end = '\n')
        print("predict is : " ,self.y_pred_svm_test)
        print( end = '\n')

        print("accuracy is : " ,svm_accuracy)
        print( end = '\n')
        print("confusion matrix is : " ,Confusion_Matrix_svm)
        print( end = '\n')
   
        print ("----------------------------------------------------------------")
        
    #load model    
    def load_svm(self,test1):
        svm_model = joblib.load(self.model2_filename)
        self.result = svm_model.predict(test1)
        print( end = '\n')
        print("Svm predict is : " , self.result)
        print( end = '\n')
        print(".................................................................")
        
        
        


    






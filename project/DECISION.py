
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt # for data visualization 
from sklearn.metrics import confusion_matrix
import joblib


class decision_tree:
    def __init__ (self ,x_train , x_test , y_train ,  y_test , model_decision_filename ,result , y_pred_decision_test,model_1):
        self.x_train =x_train 
        self.x_test = x_test
        self. y_train= y_train 
        self.y_test=y_test
        self.model_decision_filename = "decision_model_job_lib.job_lib" 
        self.result= []
        self.y_pred_decision_test= []
        self.model_1 = DecisionTreeClassifier(max_depth=2, random_state=0)
        
        
        
    def classification_decision (self):
            
        
        self.model_1.fit(self.x_train, self.y_train)
        
      


           # predicting train result
        y_pred_decision_train =  self.model_1.predict(self.x_train)
    
    # predicting test result
        self.y_pred_decision_test =  self.model_1.predict(self.x_test)
       
        
    
        
       # Making the Confusion Matrix for Logistic Regression
        decision_cm = confusion_matrix(self.y_test, self.y_pred_decision_test)
   # model accuracy for each model
        decisiontree_accuracy = metrics.accuracy_score(self.y_test , self.y_pred_decision_test)
           
           
           # save model 
        joblib.dump(  self.model_1, self.model_decision_filename)
        print('Model is saved into to disk successfully')
        
        print (self.y_pred_decision_test , self.model_1 , decisiontree_accuracy ,  decision_cm )
        
           
     #load model    
    def load_decision(self):
        decision_model = joblib.load(self.model_decision_filename)
        self.result = decision_model.predict(self.x_test)
        print(self.result)    
        print ("Decision tree Accuracy : " ,decision_model.score(self.x_test,self.y_test))
   




#graph

    
            
            
        
        


        
                  
           



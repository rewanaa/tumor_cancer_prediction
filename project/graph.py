import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



class Graph():
    def __init__ (self,df):
        self.df = df
        
    def pca(self):
        
        
        self.df = pd.read_csv("Tumor Cancer Prediction_Data.csv", index_col = 'Index')

        warnings.filterwarnings('ignore')
        self.df['diagnosis'] = self.df['diagnosis'].map({'M': 1, 'B':0})

        X = self.df.drop('diagnosis',axis = 1)
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        X_pca_scaled = pca.fit_transform(X_scaled)
        print('Projecting %d-dimensional data to 2D' % X_scaled.shape[1])

        plt.figure(figsize=(12,10))
        plt.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1], c=self.df['diagnosis'], alpha=0.7, s=40);
        plt.colorbar()
        plt.title('PCA projection')
        plt.style.use('seaborn-muted');
        
        
        
        
        
        
        
    

    





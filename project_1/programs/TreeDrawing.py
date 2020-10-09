# Load libraries 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier 
 

col_names = ['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor',	'firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 

# load dataset 
pima = pd.read_csv("new_data.csv", 
header=None, names=col_names) 

pima = pima.iloc[1:] 
pima.head() 

feature_cols = ['firstBlood','firstTower','firstInhibitor',	'firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
X = pima[feature_cols] # Features 
y = pima.winner # Target variable 


clf = DecisionTreeClassifier(max_depth=15,splitter='random',criterion= 'entropy') 
 
clf = clf.fit(X,y) 
 

from sklearn.tree import export_graphviz 
from six import StringIO   
from IPython.display import Image   
import pydotplus 
import os      
os.environ["PATH"] += os.pathsep + 'F:/bin/' 

dot_data = StringIO() 
export_graphviz(clf, out_file=dot_data,   
                filled=True, rounded=True, 
                special_characters=True,feature_names = 
feature_cols,class_names=['1','2']) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())   
graph.write_png('tree1.png') 
Image(graph.create_png()) 
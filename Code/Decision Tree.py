import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

#display all rows/columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Read data files and set variables
colnames=['Row_ID', 'Play_ID', 'Game_ID', 'Play_Num', 'TeamID_For', 'TeamID_Against', 'Event', 'SecondaryType', 'X_Location', 'Y_Location', 'Period', 'Period_Type', 'Period_Time', 'Time_Remaining', 'Date', 'GoalsAway', 'GoalsHome', 'Description', 'STX', 'STY', 'Rink_Side', 'Result']
playsfile = pd.read_csv('shotgoal_data.csv', names=colnames, skiprows=1)
playinfo = pd.DataFrame(playsfile)
print(playsfile.head())

#Attempt to normalize the data on 1 side of the rink
playsfile[playsfile.X_Location > 50].min()

#explore the dataset
print(playsfile.shape[0])
print(playsfile.shape[1])
print(playsfile.info())
print(playsfile.describe(include='all'))
secondarytype = {'Backhand': 1,'Deflected': 2, 'Slap Shot': 3, 'Snap Shot': 4, 'Tip-In': 5, 'Wrap-around': 6, 'Wrist Shot': 7}
playsfile.SecondaryType = [secondarytype[item] for item in playsfile.SecondaryType]
print(playsfile.head())

#initial graphing to get an idea of what we are looking at

#Scatter Plot (not good!)
plt.scatter(playinfo.x, playinfo.y)
plt.show()

#Heat Map of goal coordinates
sns.set_style("white")
sns.kdeplot(playinfo.x, playinfo.y)
sns.kdeplot(playinfo.x, playinfo.y, cmap="Reds", shade=True)
plt.show()

features=['X_Location', 'Y_Location', 'SecondaryType', 'TeamID_For']

#Attempt to remove location to see if that helps validation
#features=[ 'SecondaryType', 'TeamID_For']

#Set our Features and Targets
X=playsfile[features]
y=playsfile.Result

#Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Use PCA Feature Selection
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)

print("Coefficient Magnitude:")
print(pca.explained_variance_ratio_)

# Create decision tree
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
#Response Prediction
y_pred = clf.predict(X_test)

# First Run Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

#Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Second Run Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Confusion Matrix Validation
conf_matrix = confusion_matrix(y_test, y_pred)
Play_IDs = playinfo.Result.unique()
df_cm = pd.DataFrame(conf_matrix, index=Play_IDs, columns=Play_IDs )

#Plot Confusion Matrix
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=Play_IDs, xticklabels=Play_IDs)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import timeit


#Create your df here:
dfLoad = pd.read_csv('profiles.csv')
df = dfLoad
print(df.columns)


#Exploring body_type
df.body_type = df.body_type.replace("rather not say", np.nan)
df = df.dropna(subset=['body_type', 'drinks', 'smokes', 'drugs', 'sex', 'age', 'height'])
df = df.reset_index(drop=True)
print(df.body_type.value_counts(dropna=False))
body_type_mapping = {'jacked':0, 'athletic':1, 'fit': 2, 'thin':3, 'skinny': 4, 'average':5, 'used up':6, 'curvy':7, 'a little extra':8, 'full figured':9, 'overweight':10}
df["body_type_code"] = df["body_type"].map(body_type_mapping)
print(df["body_type_code"].head())
print(df["body_type_code"].value_counts())
#Plot the probability density of body type code
fig1 = plt.figure(constrained_layout=True)
plt.hist(df.body_type_code, bins=11, range=(0, 11), rwidth=0.8, density=True)
plt.ylabel("Probability")
plt.title("Body Type")
plt.xlim(0, 11)
plt.xticks(np.add(range(11),0.5), list(body_type_mapping.keys()), rotation='vertical')
plt.savefig("11_body_types.png", dpi=150)
plt.show()


#Explore diet
print(df.diet.value_counts(dropna=False))
#Let´s just separate vegans & vegetarians from everybody else (including NaNs)
is_veg = []
x = list(range(11))
y = np.zeros(11)
for i in range(len(df)):
    if 'veg' in str(df.diet[i]):
        is_veg.append(1)
        #How many "veg" in each body_type (used for plotting)
        y[df.body_type_code[i]] += 1
    else:
        is_veg.append(0)
df["is_veg"] = pd.Series(is_veg)
#Check if everything was ok with conversion
print(df[["diet","is_veg"]].tail(20))
#Plot the frequency of "veg" in each body_type
fig2 = plt.figure(constrained_layout=True)
y = np.divide(y,len(df)) #To get probabilities
plt.bar(x,y)
plt.xticks(range(11), list(body_type_mapping.keys()), rotation='vertical')
plt.title("Probability of (mostly) vegans, vegetarians")
plt.ylabel("Probability")
plt.savefig("No of Vegans.png", dpi=150)
plt.show()


#Explore drinks
print(df.drinks.value_counts())
#df = df.dropna(subset=['drinks'])
#df = df.reset_index(drop=True)
drink_mapping = {"not at all":0, "rarely":1, "socially":2, "often":3, "very often":4, "desperately":5}
df["drinks_code"] = df.drinks.map(drink_mapping)
#Let's see how many drinking types for each body type
xy = np.zeros((6,11))
for i in range(len(df)):
    xy[int(df.drinks_code[i])][df.body_type_code[i]] += 1
fig3 = plt.figure(constrained_layout=True)
xy = np.divide(xy, len(df))
p0 = plt.bar(x, xy[0])
p1 = plt.bar(x, xy[1], bottom=np.sum(xy[0:1], axis=0))
p2 = plt.bar(x, xy[2], bottom=np.sum(xy[0:2], axis=0))
p3 = plt.bar(x, xy[3], bottom=np.sum(xy[0:3], axis=0))
p4 = plt.bar(x, xy[4], bottom=np.sum(xy[0:4], axis=0))
p5 = plt.bar(x, xy[5], bottom=np.sum(xy[0:5], axis=0))
plt.title("Distribution of drinking habits vs. body type")
plt.ylabel("Probability")
plt.xticks(range(11), list(body_type_mapping.keys()), rotation='vertical')
legend_names = list(drink_mapping.keys())
legend_names.reverse()
plt.legend((p5[0], p4[0], p3[0], p2[0], p1[0], p0[0]), legend_names)
plt.savefig("Drinking distribution.png", dpi=150)
plt.show()


#Explore smoking
print(df["smokes"].value_counts())
#df = df.dropna(subset=["smokes"])
#df = df.reset_index(drop=True)
smokes_mapping = {"no":0, "sometimes":1, "when drinking":1, "yes":2, "trying to quit":2}
df["smokes_code"] = df["smokes"].map(smokes_mapping)
xy = np.zeros((3,11))
for i in range(len(df)):
    xy[int(df.smokes_code[i])][int(df.body_type_code[i])] += 1
xy = np.divide(xy, len(df))
fig4 = plt.figure(constrained_layout=True)
p0 = plt.bar(x, xy[0])
p1 = plt.bar(x, xy[1], bottom=np.sum(xy[0:1], axis=0))
p2 = plt.bar(x, xy[2], bottom=np.sum(xy[0:2], axis=0))
plt.title("Distribution of smoking habits vs. body type")
plt.ylabel("Probability")
plt.xticks(range(11), list(body_type_mapping.keys()), rotation='vertical')
plt.legend((p2[0], p1[0], p0[0]), ["yes", "sometimes", "no"])
plt.savefig("Smoking habits.png", dpi=150)
plt.show()


#Explore sex
print(df["sex"].value_counts())
sex_mapping = {"m":0, "f":1}
df["sex_code"] = df["sex"].map(sex_mapping)
xy = np.zeros((2,11))
for i in range(len(df)):
    xy[int(df.sex_code[i])][int(df.body_type_code[i])] += 1
xy = np.divide(xy, len(df))
fig5 = plt.figure(constrained_layout=True)
p0 = plt.bar(x, xy[0])
p1 = plt.bar(x, xy[1], bottom=np.sum(xy[0:1], axis=0))
plt.title("Male/female vs. body type")
plt.ylabel("Probability")
plt.xticks(range(11), list(body_type_mapping.keys()), rotation='vertical')
plt.legend((p1[0], p0[0]), ["Females", "Males"])
plt.savefig("Sex distribution.png", dpi=150)
plt.show()


#Explore drugs
print(df["drugs"].value_counts())
drugs_mapping = {"never":0, "sometimes":1, "often":2}
df["drugs_code"] = df["drugs"].map(drugs_mapping)
xy = np.zeros((3,11))
for i in range(len(df)):
    xy[int(df.drugs_code[i])][int(df.body_type_code[i])] += 1
xy = np.divide(xy, len(df))
fig6 = plt.figure(constrained_layout=True)
#Let's join the sometimes and yes for plotting
plt.bar(x, np.sum(xy[1:3], axis=0))
plt.title("Drugs users vs. body type")
plt.ylabel("Probability")
plt.xticks(range(11), list(body_type_mapping.keys()), rotation='vertical')
plt.savefig("Drugs distribution.png", dpi=150)
plt.show()


#Normalize the data
feature_data = df[['age', 'height','sex_code', 'drinks_code', 'smokes_code', 'drugs_code', 'is_veg']]
x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
labels = df['body_type_code']

#Split the data
data_train, data_test, labels_train, labels_test = train_test_split(feature_data, labels, test_size=0.2, random_state=42)

#Print the expected probabilities for random placement
print('\nExpected random probabilities:')
print(df['body_type'].value_counts()/len(df))
    
#Classify with Naive Bayes
classifierMNB = MultinomialNB()
classifierMNB.fit(data_train, labels_train)
labels_predict = classifierMNB.predict(data_test)
print("\nMultinomialNB accuracy:")
print(accuracy_score(labels_test, labels_predict))
print("\nMultinomialNB precision:")
print(precision_score(labels_test, labels_predict, average=None))
print("\nMultinomialNB recall:")
print(recall_score(labels_test, labels_predict, average=None))
print("\nMultinomialNB f1 score:")
print(f1_score(labels_test, labels_predict, average=None))

#Classify with K Nearest Neighbors
classifierKNN = KNeighborsClassifier(n_neighbors=5)
classifierKNN.fit(data_train, labels_train)
labels_predict = classifierKNN.predict(data_test)
print("\nK-Nearest Neighbors accuracy (k=5):")
print(accuracy_score(labels_test, labels_predict))
print("\nK-Nearest Neighbors precision (k=5):")
print(precision_score(labels_test, labels_predict, average=None))
print("\nK-Nearest Neighbors recall (k=5):")
print(recall_score(labels_test, labels_predict, average=None))
print("\nK-Nearest Neighbors f1 score (k=5):")
print(f1_score(labels_test, labels_predict, average=None))
#For various k values
k_max = 50
y_accuracy = []
y_precision = []
for k in range(1,k_max+1):
    classifierKNN = KNeighborsClassifier(n_neighbors=k)
    classifierKNN.fit(data_train, labels_train)
    labels_predict = classifierKNN.predict(data_test)
    y_accuracy.append(accuracy_score(labels_test, labels_predict))
    y_precision.append(list(precision_score(labels_test, labels_predict, average=None)))
results = pd.DataFrame(y_precision, columns=body_type_mapping.keys())
fig6 = plt.figure(constrained_layout=True)
plt.plot(list(range(1,k_max+1)), y_accuracy, 'k.-')
plt.title("K Nearest Neighbors - Accuracy vs. K")
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.savefig("KNN-11bodytypes-Accuracy.jpg", dpi=150)
plt.show()
fig7 = plt.figure(constrained_layout=True)
for body_type in body_type_mapping.keys():
    plt.plot(list(range(1,k_max+1)), results[body_type], '.-')
plt.title("K Nearest Neighbors - Precision vs. k")
plt.ylabel("Precision")
plt.xlabel("k")
plt.xlim(0, 72)
plt.legend(body_type_mapping.keys())
plt.savefig("KNN-11bodytypes-Precision.png", dpi=150)
plt.show()

#Classify with Suport Vector Machine
classifierSVC = SVC(kernel='rbf', gamma = 10, C = 5)
classifierSVC.fit(data_train, labels_train)
labels_predict = classifierSVC.predict(data_test)
print("\nSVC accuracy (gamma=10, C=5):")
print(accuracy_score(labels_test, labels_predict))
print("\nSVC precision (gamma=10, C=5)):")
print(precision_score(labels_test, labels_predict, average=None))
print("\nSVC recall (gamma=10, C=5)):")
print(recall_score(labels_test, labels_predict, average=None))
print("\nSVC f1 score (gamma=10, C=5)):")
print(f1_score(labels_test, labels_predict, average=None))

#Reducing the classification classes - does simplifying help?
body_type_mapping = {'jacked':0, 'athletic':0, 'fit': 0, 'thin':1, 'skinny': 1, 'average':2, 'used up':2, 'curvy':3, 'a little extra':3, 'full figured':3, 'overweight':3}
body_type_names = ['sporty', 'thin', 'average', 'above average']
df["body_type_code"] = df["body_type"].map(body_type_mapping)
print(df["body_type_code"].value_counts())
#Plot
fig8 = plt.figure(constrained_layout=True)
plt.hist(df.body_type_code, bins=4, range=(0, 4), rwidth=0.8, density=True)
plt.ylabel("Random Probability")
plt.title("Body Type")
plt.xlim(0, 4)
plt.xticks(np.add(range(4),0.5), body_type_names, rotation='vertical')
plt.savefig("4_body_types.png", dpi=150)
plt.show()

labels = df['body_type_code']

#Split the data
data_train, data_test, labels_train, labels_test = train_test_split(feature_data, labels, test_size=0.2, random_state=42)

#Expected probabilities for random placement
print('\nExpected random probabilities:')
print(df['body_type_code'].value_counts()/len(df))

#Classify with Naive Bayes
classifierMNB = MultinomialNB()
classifierMNB.fit(data_train, labels_train)
labels_predict = classifierMNB.predict(data_test)
print("\nMultinomialNB accuracy:")
print(accuracy_score(labels_test, labels_predict))
print("\nMultinomialNB precision:")
print(precision_score(labels_test, labels_predict, average=None))
print("\nMultinomialNB recall:")
print(recall_score(labels_test, labels_predict, average=None))
print("\nMultinomialNB f1 score:")
print(f1_score(labels_test, labels_predict, average=None))

#Classify with K Nearest Neighbors
classifierKNN = KNeighborsClassifier(n_neighbors=10)
classifierKNN.fit(data_train, labels_train)
labels_predict = classifierKNN.predict(data_test)
print("\nK-Nearest Neighbors accuracy (k=10):")
print(accuracy_score(labels_test, labels_predict))
print("\nK-Nearest Neighbors precision (k=10):")
print(precision_score(labels_test, labels_predict, average=None))
print("\nK-Nearest Neighbors recall (k=10):")
print(recall_score(labels_test, labels_predict, average=None))
print("\nK-Nearest Neighbors f1 score (k=10):")
print(f1_score(labels_test, labels_predict, average=None))
#For various k values
k_max = 70
y_accuracy = []
y_precision = []
y_recall = []
for k in range(1,k_max+1):
    classifierKNN = KNeighborsClassifier(n_neighbors=k)
    classifierKNN.fit(data_train, labels_train)
    labels_predict = classifierKNN.predict(data_test)
    y_accuracy.append(accuracy_score(labels_test, labels_predict))
    y_precision.append(list(precision_score(labels_test, labels_predict, average=None)))
    y_recall.append(list(recall_score(labels_test, labels_predict, average=None)))
precision_results = pd.DataFrame(y_precision, columns=body_type_names)
recall_results = pd.DataFrame(y_recall, columns=body_type_names)
#Plot accuracy
fig9 = plt.figure(constrained_layout=True)
plt.plot(list(range(1,k_max+1)), y_accuracy, 'k.-')
plt.title("K Nearest Neighbors - Accuracy vs. K")
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.savefig("KNN-4bodytypes-Accuracy.jpg", dpi=150)
plt.show()
#Plot precision
fig10 = plt.figure(constrained_layout=True)
for body_type in body_type_names:
    plt.plot(list(range(1,k_max+1)), precision_results[body_type], '.-')
plt.title("K Nearest Neighbors - Precision vs. k")
plt.ylabel("Precision")
plt.xlabel("k")
#plt.xlim(0, 72)
plt.legend(body_type_names)
plt.savefig("KNN-4bodytypes-Precision.jpg", dpi=150)
plt.show()
#Plot recall
fig11 = plt.figure(constrained_layout=True)
for body_type in body_type_names:
    plt.plot(list(range(1,k_max+1)), recall_results[body_type], '.-')
plt.title("K Nearest Neighbors - Recall vs. k")
plt.ylabel("Recall")
plt.xlabel("k")
#plt.xlim(0, 72)
plt.legend(body_type_names)
plt.savefig("KNN-4bodytypes-Recall.jpg", dpi=150)
plt.show()


#Classify with Suport Vector Machine
classifierSVC = SVC(kernel='rbf', gamma = 10, C = 5)
classifierSVC.fit(data_train, labels_train)
labels_predict = classifierSVC.predict(data_test)
print("\nSVC accuracy (gamma=100, C=10):")
print(accuracy_score(labels_test, labels_predict))
print("\nSVC precision (gamma=100, C=10):")
print(precision_score(labels_test, labels_predict, average=None))
print("\nSVC recall (gamma=100, C=10):")
print(recall_score(labels_test, labels_predict, average=None))
print("\nSVC f1 score (gamma=100, C=10):")
print(f1_score(labels_test, labels_predict, average=None))
gammas = [0.1, 0.2, 0.5, 0.8, 1, 1.5, 2, 5, 10, 20, 50, 100]
Cs = [0.1, 1, 10, 100]
y_accuracy = []
#Use above values of gamma and C for SVC - takes several hours!!!
#for c in Cs:
#    y_precision = []
#    for g in gammas:
#        classifierSVC = SVC(kernel='rbf', gamma=g, C=c)
#        classifierSVC.fit(data_train, labels_train)
#        labels_predict = classifierSVC.predict(data_test)
#        y_accuracy.append(accuracy_score(labels_test, labels_predict))
#        y_precision.append(list(precision_score(labels_test, labels_predict, average=None)))
#    precision_results = pd.DataFrame(y_precision, columns=body_type_names)
#    fig12 = plt.figure(constrained_layout=True)
#    plt.plot(list(range(len(gammas))), precision_results[body_type_names], '.-')
#    plt.xticks(range(len(gammas)), gammas)
#    plt.title("SVC - C=" + str(c))
#    plt.ylabel("Precision")
#    plt.xlabel("gamma")
#    plt.legend(body_type_names)
#    plt.savefig("SVC-c" + str(c) + ".jpg", dpi=150)
#    plt.show()
#Plot accuracy as a series
#fig13 = plt.figure(constrained_layout=True)
#plt.plot(list(range(len(gammas)*len(Cs))), y_accuracy, 'k.-')
#plt.title("SVC - Accuracy vs. gamma & C")
#plt.ylabel("Accuracy")
#plt.xlabel("index")
#plt.savefig("SVC-accuracy-C-0.1-1-10-100.jpg", dpi=150)
#plt.show()


#Regression methods
#Is peoples income dependent on their creative writing & age?
#Linear regression
df = dfLoad

#Get the total lenght of the essays
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
df[essay_cols] = df[essay_cols].replace(np.nan, '', regex=True)
df["essay_len"] = df["age"].mul(0)
for essay_col in essay_cols:
    df["essay_len"] += df[essay_col].apply(lambda x: len(x))
#Set -1 in income as NaN and drop all NaNs
df['income'] = df['income'].replace(-1, np.nan, regex=True)
df = df.dropna(subset=['age', 'income'])
df = df.reset_index(drop=True)

#Plot the scatter plots of Income vs. Age & fit them
x = df['age']
y = df['income']
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)
fig14 = plt.figure()
plt.scatter(x, y, alpha=0.01)
plt.xlabel('Age')
plt.xlim(0,75)
plt.ylabel('Income')
plt.title("Regression - Income vs. Age")
lr = LinearRegression()
lr.fit(x,y)
print('Regression R2 = ')
print(lr.score(x,y))
print(lr.coef_, lr.intercept_)
x = np.array([[0], [75]])
y = lr.predict(x)
print(y)
plt.plot(x,y,'r-')
plt.savefig("Regression - Income vs Age.png", dpi=150)
plt.show()
#Plot the scatter plots of Income vs. Essays' length & fit them
x = df['essay_len']
y = df['income']
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)
fig15 = plt.figure()
plt.scatter(x, y, alpha=0.01)
plt.xlabel('Essay Length')
plt.xlim(0,30000)
plt.ylabel('Income')
plt.title("Regression - Income vs. Essays' Length")
lr = LinearRegression()
lr.fit(x,y)
lr._residues
print('Regression R2 = ')
print(lr.score(x,y))
print(lr.coef_, lr.intercept_)
x = np.array([[0], [30000]])
y = lr.predict(x)
print(y)
plt.plot(x,y,'r-')
plt.savefig("Regression - Income vs Essays Length.png", dpi=150)
plt.show()


feature_data = df[['age','essay_len']]
labels = df['income']
#Split the data
data_train, data_test, labels_train, labels_test = train_test_split(feature_data, labels, test_size=0.0, random_state=42)  

mlr = LinearRegression()
mlr.fit(data_train,labels_train)
print('Regression R2 = ')
print(mlr.score(data_train,labels_train))
#labels_predict = mlr.predict(data_test)

#Regression using KNeighborsRegressor
regressorKNN = KNeighborsRegressor(n_neighbors=500, weights='distance')
regressorKNN.fit(data_train, labels_train)
print('K Neighbors Regression R2 = ')
print(regressorKNN.score(data_train, labels_train))

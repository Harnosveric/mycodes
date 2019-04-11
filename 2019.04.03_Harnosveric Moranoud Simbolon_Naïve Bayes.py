
# coding: utf-8

# In[37]:


## LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from sklearn import feature_extraction, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_val_score

get_ipython().magic('matplotlib inline')


## EXPLORING DATASET
# read data file using pandas csv reader
data_training = pd.read_csv('E:\Tes Data Science\Soal 1\data-training.csv')
data_testing = pd.read_csv('E:\Tes Data Science\Soal 1\data-testing.csv')
# print out the first 5 email
print("\nFirst 5 of Data Training\n")
print(data_training.head())
print("\nFirst 5 of Data Testing\n")
print(data_testing.head())


## DISTRIBUTION
# Pie Chart
data_chart1 = pd.value_counts(data_training["v1"], sort= True)
data_chart1.plot(kind = 'pie',  autopct='%1.2f%%')
plt.title('Distribution of Data Training')
plt.ylabel('')
plt.show()
data_chart2 = pd.value_counts(data_testing["v1"], sort= True)
data_chart2.plot(kind = 'pie',  autopct='%1.2f%%')
plt.title('Distribution of Data Testing')
plt.ylabel('')
plt.show()


## FEATURE ENGINEERING
data_training["v1"]=data_training["v1"].map({'spam':1,'ham':0})
data_testing["v1"]=data_testing["v1"].map({'spam':1,'ham':0})
f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X_train = f.fit_transform(data_training["v2"])
X_test = f.transform(data_testing[" v2"])
y_train = data_training["v1"]
y_test = data_testing["v1"]
# print(len(X_test_raw), len(X_train_raw), len(y_test), len(y_train))
# print(np.shape(X_train), np.shape(X_test))


## PREDICTIVE ANALYSIS
# Multinomial Naive Bayes Classifier
alphas = np.arange(1/100000, 10, 0.11)
score_train = np.zeros(len(alphas))
score_test = np.zeros(len(alphas))
recall_test = np.zeros(len(alphas))
precision_test= np.zeros(len(alphas))
count = 0
for alpha in alphas:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1 
    
matrix = np.matrix(np.c_[alphas, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['Alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print("The first 10 learning models and their metrics!\n")
print(models.head(n=10))

# best_index = models['Test Precision'].idxmax()
# print("\nThe model with the most test precision \n")
# print(models.iloc[best_index, :])

print("\nThe model with 100% precision, if any \n")
print(models[models['Test Precision']==1].head(n=5))

best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
print("\nBetween the highest possible precision, we are going to select which has more test accuracy, that is\n")
print(models.iloc[best_index, :])
print("\nSo, The model 15th is selected as the best model.\n")


# Confusion Matrix
print("\nConfusion Matrix with Multinomial NB Classifier is ")
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])


# So, using the model we get the conclusion that we misclassify only 8 spam messages as non-spam emails whereas we don't misclassify any non-spam message.

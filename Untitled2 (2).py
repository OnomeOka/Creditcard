#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.metrics import confusion_matrix 
import pickle
from sklearn.preprocessing import StandardScaler
import warnings


# In[8]:


df = pd.read_csv('creditcard.csv')


# In[9]:


df.head()


# In[11]:


df.isnull().sum()


# In[12]:


df.duplicated()


# In[15]:


# check if data target is imbalanced
print('Count of honest vs fradulent transactions')
print(df.groupby('Class')['Class'].count())

print('Proportion of faudulent transactions:',
      round(df.groupby('Class')['Class'].count()[1]/df.groupby('Class')['Class'].count()[0], 4))


# In[17]:


# Boxplot of Amount for Honest VS Fraud, excluding outliers
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax=ax1, data=df, x='Class', y='Amount', showfliers=False)
s = sns.boxplot(ax=ax2, data=df, x='Class', y='Amount', showfliers=False)
plt.show()


# In[18]:


# Correlation Plot
corr = df.corr()
sns.heatmap(data=corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=0.1, cmap='Reds')
plt.show()


# In[20]:


df[df['Class']  ==1]


# In[23]:


# Mean Transaction Amount over hour by class
df['Hour'] = df['Time'].apply(lambda x: np.floor(x/3600))
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))
df_1 = df[df['Class'] == 1]
df_0 = df[df['Class'] == 0]

s = sns.lineplot(ax=ax1,data=df_1, x='Hour', y='Amount', color='Red', errorbar=None)
s = sns.lineplot(ax=ax2,data=df_0, x='Hour', y='Amount', errorbar=None)

plt.show()



# In[24]:


# Sum of Transaction Amount over Hour by class
df['Hour'] = df['Time'].apply(lambda x: np.floor(x/3600))
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))
df_1 = df[df['Class'] == 1]
df_0 = df[df['Class'] == 0]

s = sns.lineplot(ax=ax1,data=df_1, x='Hour', y='Amount', color='Red', errorbar=None, estimator='sum')
s = sns.lineplot(ax=ax2,data=df_0, x='Hour', y='Amount', errorbar=None, estimator='sum')

plt.show()


# In[25]:


# Apply the standard scaler to the 'Time' column of the dataframe 'df'
# This will standardize the 'Time' column to have a mean of 0 and a standard deviation of 1
scale = StandardScaler()

df[['Time']] = scale.fit_transform(df[['Time']])
df[['Amount']] = scale.fit_transform(df[['Amount']])


# In[41]:


# Define the target variable, which we want to predict

target = 'Class'

# Define the predictor variables, which will be used as features for prediction
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Select the predictor variables (features) from the dataframe

X = df.loc[:, predictors]
Y = df.loc[:, target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2024)


# In[42]:


from imblearn.over_sampling import SMOTE
from collections import Counter

# Apply SMOTE to the training data to handle class imbalance
smote = SMOTE()
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
print('Resampled dataset shape %s' % Counter(Y_train_smote))


# In[43]:


train_data = lgb.Dataset(X_train_smote, label = Y_train_smote)


# In[44]:


param_grid = {
    'num_leaves': [30],
    'learning_rate': [0.05],
    'feature_fraction': [1.0]
}


# In[45]:


kf = KFold(n_splits=5, shuffle=True, random_state=2024)


# In[46]:


np.random.seed(42)
# initialise our desired values
best_score = 0
best_params = {}
models = []
scores = []

# iterating over each permutation of our param_grid
# trains models using our 5-fold cv
for num_leaves in param_grid['num_leaves']:
    for learning_rate in param_grid['learning_rate']:
        for feature_fraction in param_grid['feature_fraction']:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'is_unbalance': 'true',
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'feature_fraction': feature_fraction,
                'verbose': -1
            }

            fold_scores = []
            for train_index, val_index in kf.split(X_train_smote):
                X_train_fold, X_val_fold = X_train_smote.iloc[train_index], X_train_smote.iloc[val_index]
                Y_train_fold, Y_val_fold = Y_train_smote.iloc[train_index], Y_train_smote.iloc[val_index]

                train_data = lgb.Dataset(X_train_fold, label=Y_train_fold)
                val_data = lgb.Dataset(X_val_fold, label=Y_val_fold)

                model = lgb.train(params,
                                  train_data,
                                  valid_sets=[val_data],
                                  num_boost_round=2000, 
                                  callbacks=[lgb.early_stopping(stopping_rounds=200),]
                                 )
                
                score = model.best_score['valid_0']['auc']
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            scores.append(avg_score)
            models.append(model)

            # Update best score and best params
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

# output the best parameters and their corresponding score
print(f"Best Model Parameters: {best_params}")
print(f"Best Model AUC: {best_score}")

# use the best parameters to re-train the model on the full training set
best_model = lgb.train(best_params,
                       lgb.Dataset(X_train_smote, label=Y_train_smote),
                       num_boost_round=2000,
                       valid_sets=[lgb.Dataset(X_train_smote, label=Y_train_smote)],
                       callbacks=[lgb.early_stopping(stopping_rounds=200),])


# In[47]:


Y_pred = best_model.predict(X_test)


# In[49]:


from sklearn.metrics import  precision_score, recall_score, roc_curve, precision_recall_curve, auc


# In[50]:


precision, recall, _ = precision_recall_curve(Y_test, Y_pred)

#calculate the AUC for the precision-Recall Curve
auprc = auc(recall, precision)

print(f'Test AUPRC: {round(auprc,3)}')


# In[51]:


# Plotting
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % auprc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="best")
plt.show()


# In[53]:


# Set the decision threshold for converting probabilities to binary predictions
threshold = 0.5

# Convert predicted probabilities to binary predictions using the threshold
predictions = [1 if prob > threshold else 0 for prob in Y_pred]

cm = confusion_matrix(Y_test, predictions)

plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[54]:


precision = cm[1][1]/(cm[1][1]+cm[1][0])
print('Precision:', round(precision,3))


# In[55]:


specificity = cm[0][0]/(cm[0][0]+cm[0][1])
print('Specificity:', round(specificity,3))


# In[56]:


importances = best_model.feature_importance()
feature_names = best_model.feature_name()


# In[58]:


feature_importance_df = pd.DataFrame({
    'Feature Name': feature_names,
    'Importance': importances
})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)


# In[59]:


plt.figure(figsize=(6, 6))
sns.barplot(x='Importance', y='Feature Name', data=feature_importance_df.head(20))
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature Names')
plt.show()


# In[ ]:





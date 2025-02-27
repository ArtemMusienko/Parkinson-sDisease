import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import wget


url = 'https://storage.yandexcloud.net/academy.ai/practica/parkinsons.data'
wget.download(url)

df = pd.read_csv("./parkinsons.data")
df.info()
df.head()

features=df.loc[:,df.columns!='status'].values[:,1:] #все столбцы, кроме "status"
labels=df.loc[:,'status'].values #только "status"

print(labels[labels==1].shape[0], labels[labels==0].shape[0]) #подсчёт меток

scaler=MinMaxScaler((-1,1)) #нормализация данных
x=scaler.fit_transform(features) #подбираем данные и преобразовываем их
y=labels

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

model = XGBClassifier(learning_rate=0.3, max_depth=10, verbosity=2, random_state=42, scale_pos_weight=1.5, eval_metric='mlogloss') #инициализируем XGBClassifier

model.fit(x_train, y_train) #обучаем модель

y_pred = model.predict(x_test) #выполнение предсказания на тестовых данных
print(accuracy_score(y_test, y_pred) * 100) #точность модели
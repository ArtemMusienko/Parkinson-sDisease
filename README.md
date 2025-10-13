![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Обнаружение болезни паркинсона с помощью XGBoost
**Основная задача** - с помощью Data Science предсказать заболевание паркинсона на ранней стадии, используя алгоритм машинного обучения  **XGBoost**  и библиотеку  **sklearn**  для нормализации признаков.
Скачаем файл "**parkinsons.data**":

    url  =  'https://storage.yandexcloud.net/academy.ai/practica/parkinsons.data'
    wget.download(url)
Импортируем необходимые библиотеки:

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
Проверяем файл, узнаем его структуру и выводим его в виде таблицы:

    df = pd.read_csv("./parkinsons.data")
    df.info()
    df.head()
Перед разработкой основной части задания следует ознакомиться с описанием признаков и меток датасета, скачав файл по ссылке: [https://storage.yandexcloud.net/academy.ai/practica/parkinsons.names](https://www.google.com/url?q=https%3A%2F%2Fstorage.yandexcloud.net%2Facademy.ai%2Fpractica%2Fparkinsons.names)
Получим объекты и метки из датафрейма. Объектами являются все столбцы, **кроме** "**status**", а метки - те, что находятся в столбце **‘status’**.

    features=df.loc[:,df.columns!='status'].values[:,1:]  #все столбцы, кроме "status"
    labels=df.loc[:,'status'].values #только "status"
Столбец "**status**" имеет значения 0 и 1 в качестве меток. Получим количество этих меток:

    print(labels[labels==1].shape[0], labels[labels==0].shape[0])  #подсчёт меток
Произведем нормализацию данных с помощью **MinMaxScaler**. Масштабируем объекты от -1 до 1:

    scaler=MinMaxScaler((-1,1))  #нормализация данных
    x=scaler.fit_transform(features)  #подбираем данные и преобразовываем их
    y=labels
Теперь разделим набор данных на обучающий и тестовый наборы, сохранив 20% данных для **тестирования**:

    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
Зададим скорость обучения = 0.1, максимальная глубина дерева = 20, уровень детализации = 2, начальное значение для генератора случайных чисел = 42, вес для положительных примеров = 1.5 и используем предсказаний модели на основе **логарифмической потери**:

    model = XGBClassifier(learning_rate=0.3, max_depth=10, verbosity=2, random_state=42, scale_pos_weight=1.5, eval_metric='mlogloss')  #инициализируем XGBClassifier
Перед запуском обучения модели я поменял версию библиотеки **scikit-learn**, выполнив следующую команду "**pip install scikit-learn==1.1.3**". С версией 1.6.1 код выдавал ошибку "**super' object has no attribute '**sklearn_tags****".

    model.fit(x_train, y_train)  #обучаем модель
Определение точности работы модели:

    y_pred = model.predict(x_test)  #выполнение предсказания на тестовых данных
    print(accuracy_score(y_test, y_pred) * 100)  #точность модели

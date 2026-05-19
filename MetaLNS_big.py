# %%
path = "data_base/bbc_text_cls_adapted.csv"

# %%
import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics import confusion_matrix

# %%
data = pd.read_csv(filepath_or_buffer = path)

# %%
labels = data["labels"].unique()

# %%
X = data["text"]
y = data["labels"]

# %%
if y.dtype == object or str:
    flag = True
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = pd.DataFrame(y)

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 1, stratify = y)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# %%
indices_treino = y_train.index
columns = data.iloc[indices_treino, 2:]

# %%
columns = columns.apply(le.transform)

# %%
matrix = np.array(columns)

# %%
matrix = np.array([list(dict.fromkeys(lista)) for lista in matrix], dtype=object)

# %%
len_ = 0
for _ in matrix:
    if 1 < len(_):
        len_+=1

# %%
labels_trans = le.transform(labels)

# %%
individual = random.choices(labels_trans, k=len_)

# %%
individual = np.array(individual)

# %%
from sklearn.naive_bayes import MultinomialNB
naive_bayes_mult = MultinomialNB()

# %%
from sklearn.metrics import r2_score

# %%
def removal_function(individual,n_remove):
    # individual = list(individual)
    indexes = []
    length = len(individual)
    num = length-n_remove
    index = random.randint(0,num)
    indexes.append(index)
    indexes.append(index+1)
    indexes.append(index+2)

    return(indexes)

# %%
def determinator(tiny_list):
    choice = random.choice(tiny_list)
    return(choice)

# %%
def insertion_function(indexes,individual,matrix): 
    valores_transformados = [determinator(matrix[v]) for v in indexes]
    for index, _ in enumerate(indexes):
        individual[_] = valores_transformados[index]
        
    return(individual)
    

# %%
def make_list_proba(df_):
    total_proba_ = df_[0].prod()
    total_proba_2 = df_[1].prod()
    total_proba_3 = df_[2].prod()
    total_proba_4 = df_[3].prod()
    total_proba_5 = df_[4].prod()
    list_proba_ = []
    list_proba_.append(total_proba_)
    list_proba_.append(total_proba_2)
    list_proba_.append(total_proba_3)
    list_proba_.append(total_proba_4)
    list_proba_.append(total_proba_5)
    return(list_proba) 

# %%
def fitness_function(individual,matrix,X_train_transformed):
    matrix_ = matrix.copy()
    df_ = pd.DataFrame()
    cont = 0
    for index,values in enumerate(matrix_):
        if len(values) > 1:
            matrix_[index] = individual[cont]
            cont+=1

    matrix_ = [item[0] if isinstance(item, list) else item for item in matrix_]

    naive_bayes_mult.fit(X_train_transformed,matrix_)
    naive_bayes_mult.predict(X_train_transformed)
    df_ = naive_bayes_mult.predict_log_proba(X_train_transformed)
    list_before_sum = [max(i) for i in df_]
    fitness = sum(list_before_sum)
    
    return(fitness)

# %%
def implementation_function(individual,n_remove,X_train_transformed,matrix,k):
    best_fitness = float('-inf')
    best_individual = []
    fitness = 0
    
    for _ in range(k):
        indexes = removal_function(individual,n_remove)
        individual = insertion_function(indexes,individual,matrix)
        fitness = fitness_function(individual,matrix,X_train_transformed)
        # print(fitness)
        # print(best_fitness)
        
        if fitness > best_fitness:
            best_individual = individual
            best_fitness = fitness
        else:
            if math.exp(fitness - best_fitness) > random.random():
                best_individual = individual
                best_fitness = fitness

    return(best_fitness,best_individual)

# %%
best_fitness,best_individual = implementation_function(individual,3,X_train_transformed,matrix,1000)

# %%
print(best_fitness)
print(best_individual)

# %%
matrix_ = matrix.copy()

df_ = pd.DataFrame()
cont = 0
for index,values in enumerate(matrix_):
    if len(values) > 1:
        matrix_[index] = best_individual[cont]
        cont+=1

matrix_ = [item[0] if isinstance(item, list) else item for item in matrix_]

# %%
if flag == True:
    matrix_ = le.inverse_transform(matrix_)
    y_train = le.inverse_transform(y_train)

# %%
confusion_matrix(y_train, matrix_)

# %%




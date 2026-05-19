# %%
path = "data_base/bbc_text_cls_adapted.csv"

# %%
import pandas as pd
import numpy as np
import random
import math

# %%
data = pd.read_csv(filepath_or_buffer = path)

# %%
labels = data["labels"].unique()

# %%
X = data["text"]
y = data["labels"]

# %%
if y.dtype == object or str:
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
individual_lenght = (len(data.columns)-2)

# %%
individual = np.array([])

for i in range(individual_lenght):
    individual = np.append(individual,random.random())

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
def insertion_function(indexes,individual): 
    for index in indexes:
        individual[index] = random.random()
        
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
def determinator(tiny_list,individual):
    choice = random.choices(tiny_list, weights=individual, k=1)[0]
    return(choice)

# %%
def fitness_function(individual,X_train_transformed,matrix):
    choices = np.array([])
    choices = np.apply_along_axis(lambda x: determinator(x, individual), axis=1, arr=matrix)

    naive_bayes_mult.fit(X_train_transformed,choices)
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
        individual = insertion_function(indexes,individual)
        fitness = fitness_function(individual,X_train_transformed,matrix)
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



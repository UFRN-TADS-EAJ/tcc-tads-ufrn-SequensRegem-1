# %%
class Individual:
    def __init__(self,genome,fitness = 0,validated=False):
        self.genome = genome
        self.fitness = fitness
        self.validated = validated

# %%
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# %%
from sklearn.naive_bayes import MultinomialNB
naive_bayes_mult = MultinomialNB()
from sklearn.metrics import r2_score

# %%
path = "data_base/bbc_text_cls_adapted.csv"

# %%
data = pd.read_csv(filepath_or_buffer = path)

# %%
labels = data["labels"].unique()

# %%
X = data["text"]
y = data["labels"]
labels = y.unique()

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
matrix

# %%
individual_lenght = (len(data.columns)-2)

# %%
individual_lenght

# %%
labels = le.transform(labels)

# %%
def genome_generator(individual_lenght):
    genome = np.array([])

    for i in range(individual_lenght):
        genome = np.append(genome,random.random())
    return(genome)

# %%
def create_initial_population(size,individual_lenght,labels,matrix,X_train_transformed):
    population = []
    for _ in range(size):
        genome = genome_generator(individual_lenght)
        individual = Individual(genome=genome)
        fitness = fitness_function(individual, matrix, X_train_transformed)
        individual.fitness = fitness
        population.append(individual)
        
    return population

# %%
def determinator(tiny_list,individual):
    choice = random.choices(tiny_list, weights=individual, k=1)[0]
    return(choice)

# %%
def fitness_function(individual,matrix,X_train_transformed):
    genome = individual.genome
    choices = np.array([])
    choices = np.apply_along_axis(lambda x: determinator(x, genome), axis=1, arr=matrix)

    naive_bayes_mult.fit(X_train_transformed,choices)
    naive_bayes_mult.predict(X_train_transformed)
    df_ = naive_bayes_mult.predict_log_proba(X_train_transformed)
    list_before_sum = [max(i) for i in df_]
    fitness = sum(list_before_sum)
    
    return(fitness)

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
def selection(selected_tournment):
    selected = []
    for _ in range(len(selected_tournment)):
        first_pair = [selected_tournment[0],selected_tournment[1]]
        second_pair = [selected_tournment[2],selected_tournment[3]]
        winner_1 = max(first_pair, key=lambda x: x.fitness)
        winner_2 = max(second_pair, key=lambda x: x.fitness)
        selected.append(winner_1)
        selected.append(winner_2)
    return selected

# %%
def genome_decision(genome_1,genome_2):
    genome_1 = list(genome_1)
    genome_2 = list(genome_2)

    len_ = len(genome_1)
    
    decision_parent = []
    genome_child = []
    
    for ind in range(len_):
        decision_parent.append(random.randint(0,1))

    for ind in range(len_):
        if decision_parent[ind] == 1:
            genome_child.append(genome_1[ind])
        else:
            genome_child.append(genome_2[ind])

    return(genome_child)

# %%
def crossover(selected_1, selected_2,matrix,X_train_transformed):
    
    child_1 = Individual(genome = genome_decision(selected_1.genome,selected_2.genome))
    child_2 = Individual(genome = genome_decision(selected_2.genome,selected_1.genome))
    
    fitness_1 = fitness_function(child_1, matrix, X_train_transformed)
    fitness_2 = fitness_function(child_2, matrix, X_train_transformed)

    child_1.fitness = fitness_1
    child_2.fitness = fitness_2
    
    return child_1, child_2

# %%
def mutation(child, mutation_rate,labels,matrix,X_train_transformed):
    genome = child.genome
    child.validated = False
    
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            sorted_index = random.randrange(len(genome))
            if(genome[sorted_index]==1):
                genome[sorted_index] = random.choice(labels)
                
    child.genome = genome
    child.fitness = fitness_function(child,matrix,X_train_transformed)
    
    return(child)

# %%
def select_individuals(population):
    available = range(len(population))
    sorted_parents = random.sample(available,4)

    parent1 = population[sorted_parents[0]]
    parent2 = population[sorted_parents[1]]
    parent3 = population[sorted_parents[2]]
    parent4 = population[sorted_parents[3]]

    selected_tournment = [parent1,parent2,parent3,parent4]
    
    selected_crossover = selection(selected_tournment)

    return(selected_crossover)

# %%
# Main genetic algorithm function
def genetic_algorithm(population_size,generations, mutation_rate,labels,individual_lenght,matrix,X_train_transformed):
    population = create_initial_population(population_size,individual_lenght,labels,matrix,X_train_transformed)

    for generation in range(generations):
        new_population = []

        best_individual = max(population, key=lambda x: fitness_function(x, matrix, X_train_transformed))
        best_fitness = fitness_function(best_individual, matrix, X_train_transformed)

        index_best_individuals = sorted(range(len(population)), key=lambda x: population[x].fitness)
        primary_best_individual = population[index_best_individuals[0]]
        second_best_individual = population[index_best_individuals[1]]

        for _ in range(int((population_size-2)/2)):

            selected_crossover = select_individuals(population)
    
            child_1, child_2 = crossover(selected_crossover[0], selected_crossover[1], matrix, X_train_transformed)
    
            child_1 = mutation(child_1, mutation_rate, labels, matrix, X_train_transformed)
            child_2 = mutation(child_2, mutation_rate, labels, matrix, X_train_transformed)
            
            # sorted_index = random.sample(range(len(population)),2)
            # print(sorted_index)
            # population[sorted_index[0]] = child1
            # population[sorted_index[1]] = child2
            
            # index_worst_individuals = sorted(range(len(population)), key=lambda x: population[x].value)
            new_population.append(child_1)
            new_population.append(child_2)

        new_population.append(primary_best_individual)
        new_population.append(second_best_individual)

        population = new_population

    return (best_individual)

# %%
best_individual = genetic_algorithm(20,50,1,labels,individual_lenght,matrix,X_train_transformed)

# %%
best_individual.fitness

# %%
best_individual.genome



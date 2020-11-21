# import numpy as np
# import pandas as pd
import random


# import matplotlib.pyplot
# %matplotlib inline

# defining various steps required for the genetic algorithm
def initilization_of_population(size, n_feat):  # potential K and size
    population = [[random.randint(0, 1) for i in range(n_feat)] for j in range(size)]
    return population


xx = initilization_of_population(7, 4)
print(xx)


def target_init(arr_x):
    for j in range(len(arr_x)):
        for i in range(len(arr_x[0])):
            if random.randint(0, 1) == 1:
                arr_x[j][i] = 1
                break
            else:
                if i == len(arr_x[0])-1:
                    arr_x[j][i] = 1
    return arr_x


arr_p = [[random.randint(0, 0) for i in range(4)] for j in range(7)]
fg = target_init(arr_p)
print('fdfd', fg)


def check_pp_range(arr_pp, j, i, count):
    if arr_pp[j][i] == 1:
        count += 1
    if len(arr_pp) == i:
        if arr_pp[j][i+1] == 1:
            count += 1
    if arr_pp[j][i-1] == 1:
        count += 1
    return count

# tryr = 0
for a in range(len(fg)):
    if fg[3][a] == 1:
        tryr = a
        print('tyee',tryr)
        break

cnt = 0
gh = check_pp_range(xx, 1, tryr, cnt)
ghf = check_pp_range(xx, 0, tryr, gh)

print(gh)
print(ghf)

def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:, chromosome], y_train)
        predictions = logmodel.predict(X_test.iloc[:, chromosome])
        scores.append(accuracy_score(y_test, predictions))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds, :][::-1])


def selection(pop_after_fit, n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    population_nextgen = pop_after_sel
    for i in range(len(pop_after_sel)):
        child = pop_after_sel[i]
        child[3:7] = pop_after_sel[(i + 1) % len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen


def mutation(pop_after_cross, mutation_rate):
    population_nextgen = []
    for i in range(0, len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j] = not chromosome[j]
        population_nextgen.append(chromosome)
    # print(population_nextgen)
    return population_nextgen


def generations(size, n_feat, n_parents, mutation_rate, n_gen, X_train,
                X_test, y_train, y_test):
    best_chromo = []
    best_score = []
    population_nextgen = initilization_of_population(size, n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print(scores[:2])
        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo, best_score

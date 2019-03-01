import sys
import numpy as np
import operator
import math
import random

def euc_dist(arr1, arr2):
    diff = np.array(arr1) - np.array(arr2)
    return np.sum(np.dot(diff, diff))

def get_accident_input():
    file = open("dataset/Accident.csv", 'r')
    return np.loadtxt(file, delimiter=",", skiprows=1)

class NeuralNet():
    def __init__(self, inputs, outputs, lr):
        self.inputs = inputs
        self.outputs = outputs
        self.lr = lr

    def train(self, method, max_iter, **args):
        return method(self, max_iter, **args)


    def get_fit(self, weights):
        output = 1 / (1 + np.exp(-(np.matmul(self.inputs, weights)))) # sigmoid
        error = abs(self.outputs - output)
        return -sum(error)

    def get_best(self):
        return np.array([random.uniform(-1, 1) for i in range(13)])

    def get_neighbors(self, weight):
        n_weights = []
        for index, i in enumerate(weight):
                f_i = weight[:]
                f_i[index] = i + self.lr
                n_weights.append(f_i)
                f_i = weight[:]
                f_i[index] = i - self.lr
                n_weights.append(f_i)
        return n_weights
    

def Hill_Climbing(nn, max_iter):
    x_best = nn.get_best()
    best_fit = nn.get_fit(x_best)
    count = 0

    while True:
        old_fit = 0
        best_fit_neighbor = nn.get_neighbors(x_best)[0]
        for neighbor in nn.get_neighbors(x_best):
            fit = nn.get_fit(neighbor)
            if fit > old_fit:
                old_fit = fit
                best_fit_neighbor = neighbor
        if old_fit > best_fit:
            x_best = best_fit_neighbor
            best_fit = old_fit
        else:
            break
        count += 1
        if count > max_iter:
            break

    return x_best, best_fit

def Hill_Climbing_max_iter(nn, iterations):
    x_best, best_fit = Hill_Climbing(nn, iterations)
    for i in range(iterations):
        new_best, new_best_fit = Hill_Climbing(nn, i + 1)
        if new_best_fit > best_fit:
            x_best = new_best
            best_fit = new_best_fit
    return x_best, best_fit

def Simulated_Annealing(nn, max_iter, T0=1, tchange=.01):
    x_best = nn.get_best()
    best_fit = nn.get_fit(x_best)
    count = 0
    
    iterations = 1
    T = T0
    
    while True:
        rand_neighbor = nn.get_neighbors(x_best)[random.randint(0,len(x_best)-1)]
        fit = nn.get_fit(rand_neighbor)

        if fit > best_fit:
            x_best = rand_neighbor
            fit = best_fit
        else:
            val = math.e ** ((fit - old_fit) / temp)
            if random.uniform(0,1) < val:
                x_best = rand_neighbor
                fit = best_fit
        T -= tchange
        iterations += 1
        if iterations > n:
            break

    return x_best, best_fit

def crossover(gene1, gene2):
    pos = random.randint(0, len(gene1) - 1)
    return gene1[pos:] + gene2[:pos]

def Generic_Algorithm(nn, max_iter, init_population=[]):
    #init_population = [nn.get_best() for i in range(pop_size)]
    old_population = []
    iterations = 1

    while True:
        if old_population == population or iterations > max_iter:
            break
        fits = [nn.get_fit(x) for x in population]
        best_fits = fits.sort()[:2]
        least_fits = fits.sort()[len(fits) - 1]
        best_fit_population = [population[fits.index(x)] for x in best_fits]
        old_population = population
        crossovered = crossover(best_fit_population[0], best_fit_population[1])
        population[least_fits] = crossovered
        iterations += 1

    fits = [nn.get_fit(x) for x in population]
    best_fit = fits.sort()[0]
    best_fit_gene = population[best_fit]
    return best_fit_gene, best_fit

if __name__ == "__main__":
    print("big input")
    array = get_accident_input()
    x = array[:, :13]
    x = (x / x.max(axis=0))
    tx = x[:len(array) * 9 / 10]
    ty = array[:len(array) * 9 / 10, 13]
    vx = x[len(array) * 9 / 10:]
    vy = array[len(array) * 9 / 10:, 13]

    def HC_max_iter(inputs, outputs):
        lrs = [.001, .01, .1, .5]
        iters = [1, 5, 10, 20, 50,100]
        record = np.zeros((len(lrs), len(iters)))
        for y_count, iteration in enumerate(iters):
            for x_count, lr in enumerate(lrs):
                nn = NeuralNet(inputs, outputs, lr)
                best, fit = nn.train(Hill_Climbing, max_iter=int(iteration))
                print(y_count, x_count)
                record[x_count, y_count] = fit
        print(record)

    def SA_T(inputs, outputs):
        T_min = 1
        T_max = 100
        
        d_min = 1
        d_max = 20
        
        Ts = np.linspace(T_min, T_max, 15)
        ds = np.linspace(d_min, d_max, 15)
        print(ds, Ts)

        record = np.zeros((len(ds), len(Ts)))
        for y_count, d in enumerate(ds):
            for x_count, t in enumerate(Ts):
                
                nn = NeuralNet(inputs, outputs, .1)
                best, score = nn.train(SA, 20, T0=t, dt=d)
                record[x_count, y_count] = score
            print(y_count)
        print(record)

    HC_max_iter(tx, ty)
    SA_T(tx, ty)



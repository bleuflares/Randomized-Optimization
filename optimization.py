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
        self.outputs = inputs
        self.lr = lr

    def train(self, method, max_iter):
        return method(self, max_iter)

	
    def get_fit(self, weights):
        self.function_calls += 1
        output = 1 / (1 + numpy.exp(-(numpy.dot(numpy.array(self.inputs), weights)))) # sigmoid
        error = abs(self.outputs - output)
        return -sum(error[0])
	
    def get_best(self):
        '''generate random weights for NN'''
        return [random.uniform(-1,1) for i in range(len(self.inputs))]
	
    def get_neighbors(self, weight):
        '''gives neighbor weights'''
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

def Simulated_Annealing(nn, n, T0=1, dt=.01):
    '''simulated annealing
            n: iterations until forced break
            T0: (int) initial temperature
            dt: change in temperature
    '''

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
            val = math.e**((fit-old_fit)/T)
            if random.uniform(0,1) < val:
                x_best = rand_neighbor
                fit = best_fit
        T -= dt
        iterations += 1
        if iterations > n:
        	break

    return x_best, best_fit

if __name__ == "__main__":
	print("big input")
    array = get_accident_input()
    x = array[:, :13]
    x = (x / x.max(axis=0)).tolist()
    tx = x[:len(array) * 9 / 10]
    ty = array[:len(array) * 9 / 10, 13].tolist()
    vx = x[len(array) * 9 / 10:]
    vy = array[len(array) * 9 / 10:, 13].tolist()
    NeuralNet_depth(tx, ty, vx, vy, 9, 50, "normalized_accident")


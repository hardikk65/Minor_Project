import numpy as np



def objective_function(params):

    # TODO input params to the network and calculate output using those params

    # Objective function is the function which we are trying to minimize or maximize, In our case it can be MSE or binary Cross entropy


    pass



class ButterflyOptimizationAlgorithm:


    def __init__(self,population_size,dimensions,lower_bound,upper_bound,max_iter,alpha,c,p):

        self.population_size = population_size
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.alpha = alpha
        self.max_iter = max_iter
        self.c = c
        self.p = p

        self.weight_matrix = np.random.uniform(self.lower_bound,self.upper_bound,(self.population_size,dimensions))
        self.fitness_scores = np.zeros(self.population_size)
        self.best_butterfly = None
        self.best_fitness = None


    def calculate_scent(self):

        for bf in range(self.population_size):
            intensity = objective_function(self.weight_matrix[bf])**self.alpha
            fitness = self.c*intensity
            self.fitness_scores[bf] = fitness

            if fitness >= self.best_fitness:
                self.best_fitness = fitness
                self.best_butterfly = self.weight_matrix[bf]
    
    def update_weights(self):
        # TODO update p parameter usign chaotic maps provided in the paper

        for bf in range(self.population_size):
            r = np.random.random() 
            # Update p here using chaotic maps
            if r < self.p:
                # Global search
                new_weights = self.weight_matrix[bf] + self.fitness_scores[bf]*((r**2)*self.best_butterfly - self.weight_matrix[bf])
            else:
                # local search
                j = np.random.randint(0, self.population_size)
                new_weights = self.weight_matrix[bf] + self.fitness_scores[bf]*((r**2)*self.weight_matrix[bf] - self.weight_matrix[bf])

            self.weight_matrix[bf] = new_weights

    def optimize(self):

        # Basically for per epoch weight updation
        for iteration in range(self.max_iter):
            self.calculate_scent()
            self.update_weights()




        

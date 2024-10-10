import numpy as np




def objective_function(x):

    # TODO input params to the network and calculate output using those params

    # Objective function is the function which we are trying to minimize or maximize, In our case it can be MSE or binary Cross entropy

    
    return np.sum(x**2)



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


        self.fitness_evaluations = 0

        self.weight_matrix = np.random.uniform(self.lower_bound,self.upper_bound,(self.population_size,dimensions))
        self.fitness_scores = np.zeros(self.population_size)
        self.best_butterfly = None
        self.best_fitness = np.inf


    def calculate_scent(self):

        for bf in range(self.population_size):
            # print(objective_function(self.weight_matrix[bf]))
            intensity = np.abs(objective_function(self.weight_matrix[bf]))**self.alpha
            fitness = self.c*intensity
            self.fitness_scores[bf] = fitness

            self.fitness_evaluations += 1

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_butterfly = self.weight_matrix[bf]


    def chaotic_maps(self):

        if self.p < 0.7:
            return self.p/0.7
        
        return (10.0/3.0)*(1 - self.p)

        # return 4*self.p*(1 - self.p)
        
    
    def update_weights(self):
        # TODO update p parameter usign chaotic maps provided in the paper


        # print(self.fitness_scores)

        for bf in range(self.population_size):
            r = np.random.random() 
            # Update p here using chaotic maps
            self.p = self.chaotic_maps()

            # print('parameter : p',self.p)
            if r < self.p:
                # Global search
                
                new_weights = self.weight_matrix[bf] + self.fitness_scores[bf]*((r**2)*self.best_butterfly - self.weight_matrix[bf])
            else:
                # local search
                j = np.random.randint(0, self.population_size)
                k = np.random.randint(0, self.population_size)
                new_weights = self.weight_matrix[bf] + self.fitness_scores[bf]*((r**2)*self.weight_matrix[j] - self.weight_matrix[k])

            new_weights = np.clip(new_weights, self.lower_bound, self.upper_bound)
            self.weight_matrix[bf] = new_weights

    def optimize(self):

        # Basically for per epoch weight updation
        while self.fitness_evaluations < self.max_iter:
            self.calculate_scent()
            # print("Best butterfly:", boa.best_butterfly)
            self.update_weights()



if __name__ == '__main__':
    boa = ButterflyOptimizationAlgorithm(population_size=50,dimensions=30,lower_bound=-32,upper_bound=32,max_iter=100000,alpha = 0.04,c = 0.05,p = 0.3)
    # print(boa.weight_matrix)
    boa.optimize()


    print("Best butterfly:", boa.best_butterfly)
    print("Best fitness:", boa.best_fitness)

    

    print('Minimum value',objective_function(boa.best_butterfly))






        

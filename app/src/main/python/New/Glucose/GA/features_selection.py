#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    @author: Md. Rezwanul Haque
'''
#-------------------------
# imports
#-------------------------
from deap import base, creator
import random
import numpy as np
from deap import tools
import matplotlib.pyplot as plt
import GA.fitness_function as ff
from model.model import torch_model
from GA.config import config
#### Seed
np.random.seed(config.SEED)


"""
======================================================================================
==> filter feature selection method

    * Main difference between the filter and wrapper methods for feature selection? 
        
        -   The main differences between the filter and wrapper methods for 
            feature selection are: 
            Filter methods measure the relevance of features by their correlation with 
            dependent variable 
            while wrapper methods measure the usefulness of a subset of feature
            by actually training a model on it.
=======================================================================================
"""
class Feature_Selection_GA_Filter:
    """
        FeaturesSelectionGA
        This class uses Genetic Algorithm to find out the best features for an input model
        using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
        used for GA but it can be changed accordingly.

    """
    def __init__(self,x,y,cv_split=10,verbose=1):
        """
            Parameters
            -----------
            model : scikit-learn supported model, 
                x :  {array-like}, shape = [n_samples, n_features]
                     Training vectors, where n_samples is the number of samples 
                     and n_features is the number of features.
 
                y  : {array-like}, shape = [n_samples]
                     Target Values
            cv_split: int
                     Number of splits for cross_validation to calculate fitness.
            
            verbose: 0 or 1
        """
        self.n_features = x.shape[1]
        self.toolbox = None
        self.creator = self._create()
        self.cv_split = cv_split
        self.x = x
        self.y = y
        self.verbose = verbose
        if self.verbose==1:
            print("Model will select best features among {} features using cv_split :{}.".format(x.shape[1],cv_split))
            print("Shape od train_x: {} and target: {}".format(x.shape,y.shape))
        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = None
        self.mean_fitness = []
        self.best_fitness = []
    
    def evaluate(self,individual):
        fit_obj = ff.FitnessFunction(self.cv_split)
        np_ind = np.asarray(individual)
        if np.sum(np_ind) == 0:
            fitness = 0.0
        else:
            feature_idx = np.where(np_ind==1)[0]
            fitness = fit_obj.calculate_fitness_filter(self.x[:,feature_idx],self.y)
        
        if self.verbose == 1:
            print("Individual: {}  Fitness_score: {} ".format(individual,fitness))
        # self.mean_fitness.append(fitness)    
        return fitness,
    
    
    def _create(self):
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator
    
    def create_toolbox(self):
        """ 
            Custom creation of toolbox.
            Parameters
            -----------
                self
            Returns
            --------
                Initialized toolbox
        """
        
        self._init_toolbox()
        # return toolbox
        
    def register_toolbox(self,toolbox):
        """ 
            Register custom created toolbox. Evalute function will be registerd
            in this method.
            Parameters
            -----------
                Registered toolbox with crossover,mutate,select tools except evaluate
            Returns
            --------
                self
        """
        toolbox.register("evaluate", self.evaluate)
        self.toolbox = toolbox
     
    
    def _init_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox
        
        
    def _default_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        # toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("select", tools.selRoulette)
        toolbox.register("evaluate", self.evaluate)
        return toolbox
    
    def get_final_scores(self,pop,fits):
        self.final_fitness = list(zip(pop,fits))
    
        
    def generate(self,n_pop,ngen,cxpb=config.CROSSOVER_PROB,mutxpb=config.MUTATION_PROB,set_toolbox=False):
        
        """ 
            Generate evolved population
            Parameters
            -----------
                n_pop : {int}
                        population size
                cxpb  : {float}
                        crossover probablity
                mutxpb: {float}
                        mutation probablity
                n_gen : {int}
                        number of generations
                set_toolbox : {boolean}
                              If True then you have to create custom toolbox before calling 
                              method. If False use default toolbox.
            Returns
            --------
                Fittest population
        """
        
        if self.verbose==1:
            print("Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(n_pop,cxpb,mutxpb,ngen))
        
        if not set_toolbox:
            self.toolbox = self._default_toolbox()
        else:
            raise Exception("Please create a toolbox.Use create_toolbox to create and register_toolbox to register. Else set set_toolbox = False to use defualt toolbox")
        pop = self.toolbox.population(n_pop)
        CXPB, MUTPB, NGEN = cxpb,mutxpb,ngen

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = list(map(self.toolbox.evaluate, pop))
        
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print("-- GENERATION {} --".format(g+1))
            offspring = self.toolbox.select(pop, len(pop))
            self.fitness_in_generation[str(g+1)] = max([ind.fitness.values[0] for ind in pop])
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            weak_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, weak_ind))
            ##=================== 
            print("Best fitness from a generation: " +str(np.max(fitnesses)))
            print("Mean fitness from a generation: " +str(np.mean(fitnesses)))
            self.best_fitness.append(np.max(fitnesses))
            self.mean_fitness.append(np.mean(fitnesses))
            ##===================
            for ind, fit in zip(weak_ind, fitnesses):
                ind.fitness.values = fit
            print("Evaluated %i individuals" % len(weak_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            
                    # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.10
        if self.verbose==1:
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
    
        print("-- Only the fittest survives --")

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        # self.get_final_scores(pop,fits)
        
        return best_ind
        # return pop
    
    ## ==============================================
    ## get feature score
    def plot_feature_set_score(self,ngen):
        # print(len(self.mean_fitness))
        # print(len(self.best_fitness))
        
        plt.figure(figsize=(8,6))
        plt.title("Feature Set Fitness")
        plt.plot(self.best_fitness, 'ro-', linewidth=1, label="Best Fitness")
        plt.plot(self.mean_fitness, 'ko-', linewidth=1, label="Mean Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness value")
        plt.xlim(0, len(self.best_fitness))
        plt.legend()
        plt.tight_layout()
        plt.savefig("../output_figs/Fig_GA/GA_filter_plot_fit.pdf", dpi = 100)
        if self.verbose==1:
            plt.show()
    
"""
======================================================================================
==> wrapper feature selection method

    * What is wrapper feature selection method?

        - In wrapper methods, the feature selection process is based 
          on a specific machine learning algorithm that we are trying 
          to fit on a given dataset. 
=======================================================================================
"""
class Feature_Selection_GA_Wrap:
    """
        FeaturesSelectionGA
        This class uses Genetic Algorithm to find out the best features for an input model
        using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
        used for GA but it can be changed accordingly.

    
    """
    def __init__(self,x,y,model=None,flag=None, cv_split=10,verbose=1):
        """
            Parameters
            -----------
            model : scikit-learn supported model, 
                x :  {array-like}, shape = [n_samples, n_features]
                     Training vectors, where n_samples is the number of samples 
                     and n_features is the number of features.
 
                y  : {array-like}, shape = [n_samples]
                     Target Values
            cv_split: int
                     Number of splits for cross_validation to calculate fitness.
            
            verbose: 0 or 1
        """
        self.flag = flag
        self.x = x
        self.y = y
        self.n_features = x.shape[1]
        # self.model =  model
        if self.flag == "torch":
            self.model =  torch_model(self.n_features)
        else:
            self.model =  model
            
        self.toolbox = None
        self.creator = self._create()
        self.cv_split = cv_split
        
        self.verbose = verbose
        if self.verbose==1:
            print("Model will select best features among {} features using cv_split :{}.".format(x.shape[1],cv_split))
            print("Shape od train_x: {} and target: {}".format(x.shape,y.shape))
        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = None
        self.mean_fitness = []
        self.best_fitness = []
    
    def evaluate(self,individual):
        fit_obj = ff.FitnessFunction(self.cv_split)
        np_ind = np.asarray(individual)
        if np.sum(np_ind) == 0:
            fitness = 0.0
        else:
            feature_idx = np.where(np_ind==1)[0]
            # fitness = fit_obj.calculate_fitness(self.x[:,feature_idx], self.y, self.model, self.flag)
            fitness = fit_obj.calculate_fitness_wrapper(self.model, self.x[:,feature_idx], self.y, self.flag)
        
        if self.verbose == 1:
            print("Individual: {}  Fitness_score: {} ".format(individual,fitness))
        # self.mean_fitness.append(fitness)    
        return fitness,
    
    
    def _create(self):
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator
    
    def create_toolbox(self):
        """ 
            Custom creation of toolbox.
            Parameters
            -----------
                self
            Returns
            --------
                Initialized toolbox
        """
        
        self._init_toolbox()
        # return toolbox
        
    def register_toolbox(self,toolbox):
        """ 
            Register custom created toolbox. Evalute function will be registerd
            in this method.
            Parameters
            -----------
                Registered toolbox with crossover,mutate,select tools except evaluate
            Returns
            --------
                self
        """
        toolbox.register("evaluate", self.evaluate)
        self.toolbox = toolbox
     
    
    def _init_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox
        
        
    def _default_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        # toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("select", tools.selRoulette)
        toolbox.register("evaluate", self.evaluate)
        return toolbox
    
    def get_final_scores(self,pop,fits):
        self.final_fitness = list(zip(pop,fits))
        
    
    def generate(self,n_pop,ngen,cxpb=config.CROSSOVER_PROB,mutxpb=config.MUTATION_PROB,set_toolbox = False):
        
        """ 
            Generate evolved population
            Parameters
            -----------
                n_pop : {int}
                        population size
                cxpb  : {float}
                        crossover probablity
                mutxpb: {float}
                        mutation probablity
                n_gen : {int}
                        number of generations
                set_toolbox : {boolean}
                              If True then you have to create custom toolbox before calling 
                              method. If False use default toolbox.
            Returns
            --------
                Fittest population
        """
        
        if self.verbose==1:
            print("Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(n_pop,cxpb,mutxpb,ngen))
        
        if not set_toolbox:
            self.toolbox = self._default_toolbox()
        else:
            raise Exception("Please create a toolbox.Use create_toolbox to create and register_toolbox to register. Else set set_toolbox = False to use defualt toolbox")
        pop = self.toolbox.population(n_pop)
        CXPB, MUTPB, NGEN = cxpb,mutxpb,ngen

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = list(map(self.toolbox.evaluate, pop))
        
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print("-- GENERATION {} --".format(g+1))
            offspring = self.toolbox.select(pop, len(pop))
            self.fitness_in_generation[str(g+1)] = max([ind.fitness.values[0] for ind in pop])
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            weak_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, weak_ind))
            ##=================== 
            print("Best fitness from a generation: " +str(np.max(fitnesses)))
            print("Mean fitness from a generation: " +str(np.mean(fitnesses)))
            self.best_fitness.append(np.max(fitnesses))
            self.mean_fitness.append(np.mean(fitnesses))
            ##===================
            for ind, fit in zip(weak_ind, fitnesses):
                ind.fitness.values = fit
            print("Evaluated %i individuals" % len(weak_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            
                    # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.10
        if self.verbose==1:
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
    
        print("-- Only the fittest survives --")

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        # self.get_final_scores(pop,fits)
        
        return best_ind
        # return pop
    
    def plot_feature_set_score(self,ngen):
        # print(len(self.mean_fitness))
        # print(len(self.best_fitness))
        
        plt.figure(figsize=(8,6))
        plt.title("Feature Set Fitness")
        plt.plot(self.best_fitness, 'ro-', linewidth=1, label="Best Fitness")
        plt.plot(self.mean_fitness, 'ko-', linewidth=1, label="Mean Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness value")
        plt.xlim(0, len(self.best_fitness))
        plt.legend()
        plt.tight_layout()
        plt.savefig("../output_figs/Fig_GA/GA_wrapper_plot_fit.pdf", dpi = 100)
        if self.verbose==1:
            plt.show()
    

# -*-coding: utf-8 -
'''
    @author: Md. Rezwanul Haque
'''
class config:
    #---------------------------------
    # fixed params: for seed 
    #---------------------------------
    SEED        =   42

    #---------------------------------
    # model hyperparameter 
    #---------------------------------
    NUM_EPOCHS  =   100
    BATCH_SIZE  =   32
    N_SPLITS    =   10

    #---------------------------------
    # GA hyperparameter (from GA.config import config)
    #---------------------------------
    POPULATION_SIZE =   50
    GENERATION_SIZE =   50
    CROSSOVER_PROB  =   0.10
    MUTATION_PROB   =   0.2
    
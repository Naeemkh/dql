import random 
import pickle
import datetime
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation


from domain import Domain
from models import DQNLinear, DQNCNN



def create_random_domain(domain_shape, domain_type, num_wall, num_storage,
     num_gold, domain_number):
    """
    Creates random domain based on the requested numbers.
    
    Input:
        | domain_shape: (h,w)
        | domain_type: "1D", "2D" (use 2D for CNN)
        | num_wall: number of blocks
        | num_storage: number of storages
        | num_gold: number of golds
        | domain_number: A random integer to distinguish two domains with same 
        features in different location. 

    Output:
        | Generated domain.

    """
                    
    my_domain = Domain(domain_shape, domain_type, domain_number)
    
    available_loc = [[i,j] for i in range(domain_shape[0])\
         for j in range(domain_shape[1])]

    # shuffle the list of available locations
    random.shuffle(available_loc)

    walls = []
    storages = []
    golds=[]
        
    # assign walls
    for _ in range(num_wall):
        tmp = available_loc.pop()
        # my_domain.add_wall([tmp])
        walls.append(tmp)
    
    # assign storages
    for _ in range(num_storage):
        tmp = available_loc.pop()
        # my_domain.add_storage([tmp])
        storages.append(tmp)
    
    # assign golds
    for _ in range(num_gold):
        tmp = available_loc.pop()
        # my_domain.add_gold([tmp])
        golds.append(tmp)
    
    agent = [available_loc.pop()]
    
    my_domain.build_domain([walls, agent, storages, golds])
    my_domain.generate_domain_name()  

    return my_domain



class ResultsData:
    def __init__(self, episods, epsilon_history, scores_history, max_value, t_time):
        self.episods = episods
        self.epsilon_history = epsilon_history
        self.scores_history = scores_history
        self.max_value = max_value
        self.training_time = t_time


def save_results_data(eposids, epsilon_history, scores_history, max_value,
    t_time, output_folder, domain_name):
    
    results = ResultsData(eposids, epsilon_history, scores_history, max_value,
     t_time)
    # timestamp = datetime.datetime.now().strftime("%Y%m%d%I%M%S")
    with open(f'{output_folder}/{domain_name}_results.pkl', 'wb') as b:
            pickle.dump(results,b)

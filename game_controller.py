import os
import sys 
import time
import random
import string
import datetime
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from domain import Domain
from database import DataBase
from models import DQNAgent, DDQNAgent
from util import create_random_domain, save_results_data


class GameController:
    def __init__(self, domain_pr, ql_pr, dqn_pr, output_folder):
        self.domain_params = domain_pr
        self.ql_params = ql_pr
        self.dqn_params = dqn_pr
        self.output_folder = output_folder
        self._create_output_folder()
        self._initiate_domain()
        self._initiate_agent()
        self.db = DataBase()
        self.db.initialize(output_folder, 'state_action.db')
    
    def _create_output_folder(self):
        cwd = os.getcwd()
        if not os.path.isdir(os.path.join(cwd,self.output_folder)):
            os.mkdir(os.path.join(cwd,self.output_folder))

    def _initiate_domain(self):
        if self.domain_params.get('use_available_domain',None):
            self.domain = Domain(self.domain_params['domain_shape'],
             self.domain_params['domain_type'], self.domain_params["domain_number"])

            self.domain.generate_domain_name()
            stat = self.domain.load_domain(self.output_folder, self.domain_params)
            self.new_agent_loc = self.domain_params['new_agent_loc']
            
            if stat:
                print("Using available domain ...")
                return
            else:
                print("Domain is not available")
        

        print("Creating a new random domain ... ")
        self.domain = create_random_domain(self.domain_params['domain_shape'],
             self.domain_params['domain_type'], 
             self.domain_params['num_wall'], 
             self.domain_params['num_storage'],
             self.domain_params['num_gold'],
             self.domain_params["domain_number"])
        self.save_domain()
        self.domain.plot_domain(self.output_folder, self.output_folder)
        self.new_agent_loc = self.domain_params['new_agent_loc']

    def _initiate_agent(self):

        gm = self.ql_params['gamma']
        ep_max = self.ql_params['epsilon_max']
        ep_min = self.ql_params['epsilon_min']
        ep_dec = self.ql_params['epsilon_dec']
        input_dims = self.domain.get_state().shape
        n_actions = self.domain.n_actions()
        mem_s = self.dqn_params['mem_size']
        batch_s = self.dqn_params['batch_size']
        algo = self.dqn_params['algorithm']
        env_name = self.domain.domain_name
        lr = self.dqn_params['learning_rate']
        replace_every = self.dqn_params['replace_network_every']
        domain_type = self.domain_params['domain_type']

        # type of agent: 
        # DQN: Sequential Deep Qlearning
        # DDQN: Sequential Double Deep Qlearning

        if algo == "DQN":
            self.agent = DQNAgent(gamma=gm, epsilon=ep_max, lr=lr,
                    input_dims=input_dims, n_actions=n_actions,
                    mem_size=mem_s, batch_size=batch_s, domain_type=domain_type,
                    eps_min=ep_min, eps_dec=ep_dec, replace= replace_every,
                    algo=algo, env_name=env_name,
                    chkpt_dir=self.output_folder)

        elif algo == "DDQN":
            self.agent = DDQNAgent(gamma=gm, epsilon=ep_max, lr=lr,
                    input_dims=input_dims, n_actions=n_actions,
                    mem_size=mem_s, batch_size=batch_s, domain_type=domain_type,
                    eps_min=ep_min, eps_dec=ep_dec, replace= replace_every,
                    algo=algo, env_name=env_name,
                    chkpt_dir=self.output_folder)

        
        if self.dqn_params.get('load_pretrained_agent',None):
            if self.dqn_params.get('start_from', None):
                self.agent.load_models(self.dqn_params['start_from'])
                return
            self.agent.load_models()
    
    def save_agent(self):
        self.agent.save_models()

    def load_agent(self):
        self.agent.load_models()
        
    def save_domain(self):
        self.domain.save_domain(self.output_folder,
         self.domain_params)

    def show_parameters(self):
        
        print("Domain parameters:")
        for i, val in self.domain_params.items():
            print(i,":\t",val)
        
        print("\nQlearning parameters:")
        for i, val in self.ql_params.items():
            print(i,":\t",val)

        print("\nDeep Neural Network parameters:")
        for i, val in self.dqn_params.items():
            print(i,":\t",val)

    def print_domain(self):
        pass

    def train(self):
        
        best_score = -np.inf
        scores = []
        eps_history = []
        figure_number = 1

        if self.dqn_params["load_pretrained_agent"]:
            pass

        n_games = self.ql_params['n_game']

        k = 1
        episods=[]
        start_time = time.time()
        for i in range(n_games):
            done = False
            score = 0
            self.domain.reset(new_start_loc=self.new_agent_loc)
            observation = self.domain.get_state()
            jj = 1
            episods.append(k)
            k = k + 1
            while not done:       
                action = self.agent.choose_action(T.unsqueeze(T.tensor(observation, dtype=T.float32), dim=0))
                # print("This is action ===> ", action)
                new_observation, reward, done = self.domain.step(action)
                # print("========OOOO=========")
                # print(new_observation, reward, done)
                # print(self.domain.total_golds)
                # print("========OOOO=========")
                if done:
                    print("Terminated at: ",jj,"-->",done)

                # new_observation = mydomain.get_state()
                score += reward
                # if not load_checkpoint:
                self.agent.store_transition(observation, action, reward, new_observation, int(done))
                self.agent.learn()

                # my_agent.remember(observation, action, reward,new_observation, done)
                observation = new_observation
                # mydomain.plot_domain(True)
                # print(reward)
                # print("Observation ====================")
                # print(observation)
                # print("================================")
                # print('xx')
                
                # if jj % 5 == 0 and i % 50 == 0 and plotit:
                #     # print (f"jj: {jj} and i: {i}")
                #     current_state = mydomain.get_state()
                #     with T.no_grad():
                #         tmp_probs = my_agent.q_next.forward(current_state)
                #         a = tmp_probs.max(dim=0)[0]
                #     save_figure(mydomain.plot_domain(False),domain_shape,tmp_probs,a,mydomain.actions,figure_number)
                #     figure_number += 1 
                #     plotit = False

                jj += 1
                if jj > self.ql_params['stop_game_after']:
                    jj = 0
                    break
                # print("Score ==> ", score)
                # print(f"Action: {mydomain.actions[a]}, reward: {reward}")
                # if i > 1000:
                #     im = plt.imshow(mydomain.plot_domain(False), interpolation=None)        
                #     ims.append([im])
                
            # if i> 1000:
            #     ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
            #     ani.save(f'animation_test_{k}.gif', writer='imagemagick', fps=4)
            #     k = k+1
            eps_history.append(self.agent.epsilon)
            scores.append(score)
            ave_score = np.mean(scores[max(0,i-100):(i+1)])
            best_score = max(best_score, score)
            print(f'episode: {i},'
             f' agent_loc: ({self.domain.original_domain_location[1][0][0]},'
             f'{self.domain.original_domain_location[1][0][1]}),'
             f' epsilon: {self.agent.epsilon}, score: {ave_score},(max: {best_score})')
            
            if i % 100 == 0 and i > 0:
                self.agent.save_models()

        end_time = time.time()
        learning_time = end_time - start_time
        
        fig = plt.figure()
        ax = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)
        ax.plot(episods, eps_history, color="C0")
        ax.set_xlabel("Training Steps", color="C0")
        ax.set_ylabel("Epsilon", color="C0")
        ax.tick_params(axis='x', colors="C0")
        ax.tick_params(axis='y', colors="C0")
        ax2.plot(episods, scores, color="C1")
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        ax2.set_ylabel('Score', color="C1")
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(axis='y', colors="C1")

        # plt.plot(scores)
        # plt.show()

        fig.savefig(os.path.join(os.getcwd(),self.output_folder+f"/learning_score_{self.domain_params['domain_number']}.pdf"))
        save_results_data(episods,eps_history,scores,best_score,learning_time,self.output_folder,self.domain.domain_name)

        
    @staticmethod
    def generate_uid():
        """ generates 16 chars random combination from string and numbers"""
        char_list = string.ascii_uppercase + string.digits
        return ''.join(random.choice(char_list) for _ in range(16))
    
    def generate_animation_of_trained_model(self, random_agent_loc=False):
        
        # connect to database
        self.db.connect_to_db()
        
        # generate random number to track 
        random_number = self.generate_uid()
        
        done = False
        self.domain.reset(new_start_loc=random_agent_loc)
        total_reward = 0
        fig = plt.figure(101, figsize=(6, 6))
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, self.domain.ncols, 1))
        ax.set_yticks(np.arange(0.5, self.domain.nrows, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ims = []
        j = 1
        current_state = self.domain.get_state()

        while not done:
            # current state

            # predict next state
            if self.domain_params['domain_type']=='2D':
                # current_state = T.unsqueeze(current_state,0)
                current_state_expand = current_state.copy()
                current_state_expand = np.expand_dims(current_state_expand,axis=0)
                action = self.agent.predict_next_move(current_state_expand)
            else:
                action = self.agent.predict_next_move(current_state)
            
            self.db.enter_value_db(random_number, current_state, action)
            
            # take that action
            new_observation, reward, done = self.domain.step(action)
            current_state = new_observation
            # print(done)
            # reward, done = self.domain.action(self.domain.actions[a])
            
            total_reward += reward 

            im = ax.imshow(self.domain.plot_domain(False), interpolation=None)        
            ims.append([im])
            j = j + 1
            if j>400:
                break
        
        print(f'Total reward: {total_reward:0.4f}')
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
        # timestamp = datetime.datetime.now().strftime("%Y%m%d%I%M%S")
        ani.save(f'{self.output_folder}/animation_{self.domain_params["domain_number"]}_{random_number}.gif', writer='imagemagick', fps=4)
        
        print("Done with storing data into database.")
        self.db.close_db_connection()
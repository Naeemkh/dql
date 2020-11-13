import os
import string
import random
import copy
import datetime
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from domain import Domain
from models import DQNCNN
from database import DataBase
from util import create_random_domain


class CNNTrainer:
    """
    This class uses state-data to train CNN models.
    """
    agents = []
    def __init__(self, output_folder):
        self.db = DataBase()
        self.actions_list = ["Up", "Down", "Left", "Right"]
        self.output_folder = output_folder
    
    def connect_to_data_base(self, database_name):
        self.db.connect_to_db(database_name)
        self.db_connected = True

    def plot_example(self, save_to_file=False):
        uid, img, action = self.db.read_from_db(1)[0]
        img = np.moveaxis(img,[0,1,2],[2,0,1])

        fig = plt.figure(101, figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_title(f"Selected action: {str(action)} ({self.actions_list[int(action)]})")
        plt.axis('off')
        plt.imshow(img)
        
        if save_to_file:
            file_name = self.output_folder + "/sample_training_image_" +\
             str(random.randint(100,1000000)) + ".png"
            fig.savefig(file_name, dpi=64)
            return
               
        plt.show()
    

    def extract_input_data_info(self):
        if not self.db_connected:
            print("Database is not connected.")

        uid, img, action = self.db.read_from_db(1)[0]
        self.input_dims = img.shape

    def inialize_model(self, lr, n_actions, model_name):
        self.extract_input_data_info()
        self.q_model = DQNCNN(lr, n_actions,
                            input_dims=self.input_dims,
                            name = model_name,
                            chkpt_dir=self.output_folder)
        
        self.q_model.optimizer = optim.SGD(self.q_model.parameters(), lr=lr)
        self.q_model.loss = nn.CrossEntropyLoss()

        # for i in self.q_model.parameters():
        #     i.grad = True

    
    def train_it(self, n_data, n_epoch, n_batch):
        # if self.memory.mem_cntr < self.batch_size:
        #     return
        
        # extract data
        mydata = self.db.read_from_db(n_data)

        random.shuffle(mydata)
        img_org = [i[1] for i in mydata]
        action_org = [i[2] for i in mydata]

        if len(action_org) < n_batch:
            print("Not enough data for training.")
            return
        
        
        
        for i in range(n_epoch):
            action = action_org.copy()
            img = img_org.copy()

            ep_loss = 0
            ep_acc = []
            j = 0
            enough_data = True

            while enough_data:

                img_tr = img[-n_batch:]
                action_tr = action[-n_batch:]
                del img[-n_batch:]
                del action[-n_batch:]

                tmp_img = np.stack(img_tr)
                tmp_action = np.stack(action_tr)
                tmp_action = tmp_action.astype('uint8')

                # onehot_action = np.zeros((tmp_action.size, 4))
                # onehot_action[np.arange(tmp_action.size), tmp_action] = 1

                img_tr = T.from_numpy(tmp_img).type(T.FloatTensor)
                action_tr = T.from_numpy(tmp_action).type(T.FloatTensor)
                
                # img_tr = T.tensor(img_tr, requires_grad=True)
                action_tr = T.tensor(action_tr, requires_grad=True)

                self.q_model.optimizer.zero_grad()
                action_tr.to(self.q_model.device)
                img_tr.to(self.q_model.device)
                prediction = self.q_model.forward(img_tr)
                max_actions = T.argmax(prediction, dim=1)
                # max_actions = T.tensor(max_actions).type(T.long)
                action_tr = action_tr.type(T.long)
                action_tr = action_tr.to(self.q_model.device)
                loss = self.q_model.loss(prediction, action_tr).to(self.q_model.device)


                wrong = T.where(max_actions != action_tr,
                                T.tensor([1.]).to(self.q_model.device),
                                T.tensor([0.]).to(self.q_model.device))

                acc = 1 - T.sum(wrong) / n_batch
                ep_acc.append(acc.item())
                ep_loss += loss.item()
                a = list(self.q_model.parameters())[0].clone()
                loss.backward()
                self.q_model.optimizer.step()
                b = list(self.q_model.parameters())[0].clone()
                # print(T.equal(a.data, b.data))

                j = j+1

                if len(action) < n_batch:
                    enough_data = False

                # for i in self.q_model.parameters():
                #     print(i.grad)
             
            print(f'Finish epoch {str(i)}, totol loss: {ep_loss:.8f}, accuracy: {np.mean(ep_acc):.8f}')
            with open('cnn_training_results.txt','a') as fobj:
                fobj.write(f"{str(i)} {ep_loss:.8f} {np.mean(ep_acc):.8f}\n")

    def save_model(self, name):
        self.q_model.save_checkpoint(name=name)
        tmp = copy.deepcopy(self.q_model)
        self.agents.append(tmp)

    def load_model(self, model_name):
        self.inialize_model(lr = 0.0001, n_actions=4, model_name='my_model')
        model_path = os.path.join(self.output_folder,model_name)
        self.q_model.load_state_dict(T.load(model_path))
        tmp = copy.deepcopy(self.q_model)
        self.agents.append(tmp)


    
    def load_the_domain(self,domain_params):
        if domain_params.get('use_available_domain',None):
            domain = Domain(domain_params['domain_shape'],
             domain_params['domain_type'], domain_params["domain_number"])

            domain.generate_domain_name()
            stat = domain.load_domain(self.output_folder, domain_params)
            new_agent_loc = domain_params['new_agent_loc']
            
            if stat:
                print("Using available domain ...")
                return domain
            else:
                print("Domain is not available")
        

        print("Creating a new random domain ... ")
        domain = create_random_domain(domain_params['domain_shape'],
             domain_params['domain_type'], 
             domain_params['num_wall'], 
             domain_params['num_storage'],
             domain_params['num_gold'],
             domain_params["domain_number"])
        # self.save_domain()
        # self.domain.plot_domain(True, self.output_folder)
        # self.new_agent_loc = self.domain_params['new_agent_loc']
        return domain

    @staticmethod
    def predict_next_move(model, state):
        actions = model.forward(state)
        action = T.argmax(actions).item()
        return action

    @staticmethod
    def generate_uid():
        """ generates 16 chars random combination from string and numbers"""
        char_list = string.ascii_uppercase + string.digits
        return ''.join(random.choice(char_list) for _ in range(16))


    def generate_animation(self,  domain_params , random_agent_loc=False):
        
        # # connect to database
        # self.db.connect_to_db()
        
        # generate random number to track 
        random_number = self.generate_uid()
        
        
        mydomain = self.load_the_domain(domain_params)
        # mydomain.plot_domain(individual=True, into_folder=True)

        done = False
        mydomain.reset(new_start_loc=random_agent_loc)
        total_reward = 0
        fig = plt.figure(101, figsize=(6, 6))
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, mydomain.ncols, 1))
        ax.set_yticks(np.arange(0.5, mydomain.nrows, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ims = []
        j = 1
        current_state = mydomain.get_state()

        while not done:
            # current state

            # predict next state
            if domain_params['domain_type']=='2D':
                # current_state = T.unsqueeze(current_state,0)
                current_state_expand = current_state.copy()
                current_state_expand = np.expand_dims(current_state_expand,axis=0)
                tmp_action=[]
                for ag in self.agents:
                    tmp_action.append(self.predict_next_move(ag,current_state_expand))
                # action = self.agent.predict_next_move(current_state_expand)
                print(tmp_action)

            random.shuffle(tmp_action)
            action = np.argmax(np.bincount(tmp_action[:5]))
            # else:
            #     action = self.agent.predict_next_move(current_state)
            
            # self.db.enter_value_db(random_number, current_state, action)
            
            # take that action
            new_observation, reward, done = mydomain.step(action)
            current_state = new_observation
            # print(done)
            # reward, done = self.domain.action(self.domain.actions[a])
            
            total_reward += reward 

            im = ax.imshow(mydomain.plot_domain(False), interpolation=None)        
            ims.append([im])
            j = j + 1
            if j>50:
                break
        
        print(f'Total reward: {total_reward:0.4f}')
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%I%M%S")
        ani.save(f'{self.output_folder}/animation_{domain_params["domain_number"]}_{random_number}.gif', writer='imagemagick', fps=4)
        return total_reward, random_number
        # print("Done with storing data into database.")
        # self.db.close_db_connection()
    



if __name__ == "__main__":
    mycnn = CNNTrainer('cnn_database')
    mycnn.connect_to_data_base('cnn_database\\total_data.db')
    # for i in range(1):
    #     mycnn.plot_example(save_to_file=True)
    
    

    model_names = ['aa11','aa12','aa13', 'aa14', 'aa15','aa16','aa17','aa18', 'aa19', 'aa20']
    # model_names = ['aa16','aa17','aa18', 'aa19', 'aa20']
    # model_names = ['aa11','aa12','aa13', 'aa14', 'aa15']
    # for i in range(5):
    #     mycnn.inialize_model(lr = 0.0001, n_actions=4, model_name='my_model')
    #     mycnn.inialize_model(lr = 0.0001, n_actions=4, model_name='my_model')
    #     mycnn.train_it(131072, 600, 512)
    #     mycnn.save_model(model_names[i])
    
    # mycnn.inialize_model(lr = 0.0001, n_actions=4, model_name='my_model')
    for i in model_names:
        mycnn.load_model(i)
    # mycnn.load_model('aa11')
    # mycnn.load_model('aa12')
    # mycnn.load_model('aa13')
    # mycnn.load_model('aa14')
    # mycnn.load_model('aa15')

    rewards = []
    for i in range(1):
        domain_params = {'domain_shape':(10,10), "num_wall":10, "num_gold":2,
                 'num_storage':2, "use_available_domain":True, 
                 'domain_type':"2D", "new_agent_loc":True,
                 'domain_number':10165}    

        reward, rnumber = mycnn.generate_animation(domain_params=domain_params, random_agent_loc=False)
        rewards.append((reward,rnumber))
        with open('cnn_test_reward.txt','a') as fobj:
                fobj.write(f"{reward:0.8f} {str(rnumber)}\n")

    


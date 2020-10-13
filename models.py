import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, domain_type):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.set_input_shape(input_shape, domain_type)
        self.state_memory = np.zeros((self.mem_size, *self.input_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_shape),
                                          dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def set_input_shape(self, input_shape, domain_type):
        if domain_type == '1D':
            self.input_shape = (1,input_shape[0]*input_shape[1])
        else:
            self.input_shape = input_shape

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class DQNLinear(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, initial_checkpoint_file=None):
        super(DQNLinear, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.initial_checkpoint_file = initial_checkpoint_file
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims[1],200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,20)
        self.out = nn.Linear(20,4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self = self.to(self.device)

    def forward(self, x):

        x = T.tensor(x, dtype = T.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        x = T.flatten(x, start_dim = 1)

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file=None):
        
        if not checkpoint_file:
            print('... loading checkpoint ...')
            try:
                self.load_state_dict(T.load(self.checkpoint_file))
                print(f'file {self.checkpoint_file} is loaded.')
            except FileNotFoundError as e:
                print(e)
        else:
            try:
                c_file = os.path.join(self.checkpoint_dir, checkpoint_file)
                self.load_state_dict(T.load(c_file))
                print(f'file {c_file} is loaded.')
            except Exception as e:
                print(e)

class DQNCNN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, initial_checkpoint_file=None):
        super(DQNCNN, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.initial_checkpoint_file = None
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 2, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)
       
        self.fc1 = nn.Linear(fc_input_dims,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self = self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))


    def forward(self, x):
        x = T.tensor(x, dtype = T.float32)
        x = x.to(self.device)

        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        conv_state = conv3.view(conv3.size()[0],-1)
        flat0 = F.relu(self.fc1(conv_state))
        flat1 = F.relu(self.fc2(flat0))
        actions = self.fc3(flat1)

        return actions

    def save_checkpoint(self, name=None):
        print('... saving checkpoint ...')
        
        if not name:
            T.save(self.state_dict(), self.checkpoint_file)        
            # the following line is temporal. Remove it later. 
            T.save(self.state_dict(), os.path.join(self.checkpoint_dir, 'domain_10_10_init_3'))
            return

        T.save(self.state_dict(), os.path.join(self.checkpoint_dir,name))

    def load_checkpoint(self, checkpoint_file=None):        
        if not checkpoint_file:
            print('... loading checkpoint ...')
            try:
                self.load_state_dict(T.load(self.checkpoint_file))
                print(f'file {self.checkpoint_file} is loaded.')
            except FileNotFoundError as e:
                print(e)
        else:
            try:
                c_file = os.path.join(self.checkpoint_dir, checkpoint_file)
                self.load_state_dict(T.load(c_file))
                print(f'file {c_file} is loaded.')
            except Exception as e:
                print(e)
    

class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size,
                batch_size, domain_type, eps_min=0.01, eps_dec=5e-5, replace=500,
                algo = None, env_name=None, chkpt_dir = 'tmp/dqn'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.domain_type = domain_type
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, self.domain_type)

        self.name_eval = self.env_name+"_"+self.algo+"_"+str(self.lr).replace(".","p")+"_eval"
        self.name_next = self.env_name+"_"+self.algo+"_"+str(self.lr).replace(".","p")+"_next"

        
        if domain_type == "1D":
            
            self.q_eval = DQNLinear(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name = self.name_eval,
                                    chkpt_dir=self.chkpt_dir)
    
            self.q_next = DQNLinear(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name = self.name_next,
                                    chkpt_dir=self.chkpt_dir)
        elif domain_type == "2D":

                                    
            self.q_eval = DQNCNN(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name = self.name_eval,
                                    chkpt_dir=self.chkpt_dir)
    
            self.q_next = DQNCNN(self.lr, self.n_actions,
                                    input_dims=input_dims,
                                    name = self.name_next,
                                    chkpt_dir=self.chkpt_dir)


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            actions = self.q_eval.forward(observation.double())
            action = T.argmax(actions).item()
        else:
            action = T.tensor(np.random.choice(self.action_space))
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done =\
            self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            print("Deep neural network is replaced.")

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self, initial_model = None):
        if initial_model:
            self.q_eval.load_checkpoint(initial_model)
            self.q_eval.load_checkpoint(initial_model)
            return
        
        self.q_next.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # print("learned ... ")
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)


        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        # last step reward is zero.
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.q_next.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


    def predict_next_move(self, state):
        actions = self.q_eval.forward(state)
        action = T.argmax(actions).item()
        return action


class DDQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size,
                batch_size, domain_type, eps_min=0.01, eps_dec=5e-5, replace=500,
                algo = None, env_name=None, chkpt_dir = 'tmp/dqn'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.domain_type = domain_type
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, self.domain_type)

        self.name_eval = self.env_name+"_"+self.algo+"_"+str(self.lr).replace(".","p")+"_eval"
        self.name_next = self.env_name+"_"+self.algo+"_"+str(self.lr).replace(".","p")+"_next"

        
        if domain_type == "1D":
            self.q_eval = DQNLinear(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name = self.name_eval,
                                    chkpt_dir=self.chkpt_dir)
    
            self.q_next = DQNLinear(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name = self.name_next,
                                    chkpt_dir=self.chkpt_dir)

        elif domain_type == "2D":
            # input_dims = (3, self.input_dims[0], self.input_dims[1])
            self.q_eval = DQNCNN(self.lr, self.n_actions,
                                    input_dims=input_dims,
                                    name = self.name_eval,
                                    chkpt_dir=self.chkpt_dir)
    
            self.q_next = DQNCNN(self.lr, self.n_actions,
                                    input_dims=input_dims,
                                    name = self.name_next,
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):

        if np.random.random() > self.epsilon:
            actions = self.q_eval.forward(observation.double())
            action = T.argmax(actions).item()
        else:
            action = T.tensor(np.random.choice(self.action_space))
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done =\
            self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            print("Deep neural network is replaced.")

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self, initial_model = None):
        if initial_model:
            self.q_eval.load_checkpoint(initial_model)
            self.q_next.load_checkpoint(initial_model)
            return
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
       
        # last step reward is zero.
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        # q_target = (1-0.05)*q_eval[indices, max_actions] + 0.05*(rewards + self.gamma * q_next[indices, max_actions])
        # q_target = (1-0.05)*q_pred + 0.05*(rewards + self.gamma * q_next[indices, max_actions])
             

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

    def predict_next_move(self, state):
        actions = self.q_eval.forward(state)
        action = T.argmax(actions).item()
        return action
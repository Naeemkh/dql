import os
import copy
import random
import pickle
import itertools
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DAgent:
    """
    The domain agent(s) class. Controls the location and the color of the agent 
    based on different behavior or action. 
    """
    def __init__(self,agent_loc):
        self.has_box = False
        self.color = None
        self.row, self.col = agent_loc
  
    def take_action(self, item):
        """
        An agent takes action or at least tries to do so. The domain will decide
        whether it will implement the requested action or not. This is important
        because the domain will punish the action that is not valid. As a result,
        the agent, after some time of learning, should not even try to request
        an invalid action. In other words, we want to give the agent enough 
        latitude to do whatever it wants, however, it should suffer the 
        consiquences.
        """
        arow = self.row
        acol = self.col

        if item == "Up":
            arow -= 1
        elif item == "Down":
            arow += 1
        elif item == "Right":
            acol += 1
        elif item == "Left":
            acol -= 1
        else:
            print("Action is not defined.")

        return (arow, acol)

    def agent_code_color_update(self):
        # TODO: This part should be modified in case of multiagent. 
        if self.has_box:
            self.code  = 7
            self.color = [7,140,43]
        else:
            self.code = 9
            self.color = [235,7,7]

    def __repr__(self):
        return f'Agent(({self.row},{self.col}))'

    def __str__(self):
        return f'Agent in location: {self.row}, {self.col}; and color'\
               f' {self.color}' 

class Cube:
    """
    Class Cube to control the cube behavior in the domain. There are 4 different
    Cube status.
    """

    cube_values = {
        "wall" : [0, [0,0,0]],
        "fspace" : [1, [255,255,255]],
        "gold" : [2,[255,196,0]],
        "storage": [3,[255,224,224]]
    }

    def __init__(self):
        self.update("fspace")

    def update(self,ctype):
            self.ctype = ctype
            self.code = self.cube_values[ctype][0]
            self.color = self.cube_values[ctype][1]
   
    def is_available(self):
        if self.ctype == "wall":
            return False
        else:
            return True

    def get_color(self,agent):
        if not agent:
            return self.color
        else:
            agent.agent_code_color_update()
            return agent.color

    def get_code(self,agent):
        if not agent:
            return self.code
        else:
            agent.agent_code_color_update()
            return agent.code


class Domain:
    total_reward = 0
    total_golds = 0
    def __init__(self, dshape, domain_type, domain_number):
        """
        Input:
        
            | domain_type: 1D or 2D
            | dshape: tuple: (v,h): number of blocks in vertical and
             horizontal direction

            list of possibilities for reward:
            hitting_wall (n_hw) - 
            hitting_border (n_hb) -
            wandering_around (n_wa) -
            enter_cube_with_gold_while_has_gold (n_ecwg_hg) -
            attemp_to_pickup_gold_at_gold_cube_hasnot_gold (p_ecwg_hng) +
            enter_storage_while_has_gold (p_es_hg) +
            enter_storage_while_hasnot_gold (n_es_hng) -

        """
        self.nrows, self.ncols = dshape
        self.n_wall = 0
        self.n_storage = 0
        self.total_golds = 0
        self.initiate_domain()
        self.color_space_holder = np.zeros((3,self.nrows,self.ncols))
        self.code_space_holder = np.zeros((self.nrows,self.ncols))
        self.can_add_agent = True
        self.actions =  ['Up','Down','Left','Right']
        self.domain_type = domain_type
        self.original_domain_location = None
        self.domain_number = domain_number
        self.domain_name = None
   
        self.rewards = {'n_hw': -1.0,
                        'n_hb': -1.0,
                        'n_wa': -0.05,
                        'n_ecwg_hg': -0.05,
                        'p_ecwg_hng': +1.0,
                        'p_es_hg': +1.0,
                        'n_es_hng': -0.2,
                         }
        
    def initiate_domain(self):
        "Initialize the domain as n*n matrix, 1 is free space"
        self.total_golds = 0
        self.domain_mat = np.array([[Cube() for i in range(self.ncols)] for j in range(self.nrows)])
        

    def update_state(self,loc,status):
        """ loc: location of the agent.
            status: what will be.
        """
        self.domain_mat[loc[0]][loc[1]].update(status)

    def action(self,action_item):
        """
        There are 4 actions:
        - Move left 
        - Move right
        - Move up
        - Move down
        """
        
        if self.can_add_agent:
            print("There is no agent to take any action")
        else:
            
            # At this step, agent shows what it wants to do, e.g., wants to get
            #  out of the domain. However, Domain decides whether it is a valid
            #  move and give rewards accordingly. 
            (new_row, new_col) = self.agent.take_action(action_item)

            # the agent wants to change location
            if ((new_row < 0 or new_row > self.nrows-1) or
               (new_col < 0 or new_col > self.ncols-1)):
                # the agent wants to go out of the domain. Nothing will happen,
                # however, the agent will get - reward.
                this_action_reward = self.rewards["n_hb"]
            elif self.domain_mat[new_row][new_col].ctype == "wall":
                # the agent wants to go into the wall. Nothing will happen, 
                # however, the agent will get - reward.
                this_action_reward = self.rewards["n_hw"]

            elif self.domain_mat[new_row][new_col].ctype == "gold":
                # the agent wants to go into a cube with gold
                if self.agent.has_box:
                    # already has the gold, should not do this. 
                    # will recieve negative reward.
                    # self.agent.row = new_row
                    # self.agent.col = new_col
                    this_action_reward = self.rewards["n_ecwg_hg"]
                else:
                    # has not a gold, will recieve + reward
                    self.agent.row = new_row
                    self.agent.col = new_col
                    self.agent.has_box = True
                    self.domain_mat[self.agent.row][self.agent.col].update("fspace")
                    this_action_reward = self.rewards["p_ecwg_hng"]
            
            elif self.domain_mat[new_row][new_col].ctype == "fspace":
                # the agent wandering around, it is ok, however,
                # will recieve some negative rewards.
                self.agent.row = new_row
                self.agent.col = new_col
                this_action_reward = self.rewards["n_wa"]

            elif self.domain_mat[new_row][new_col].ctype == "storage":
                # the agent wandering around, it is ok, however,
                # will recieve some negative rewards.
                if self.agent.has_box:
                    self.agent.row = new_row
                    self.agent.col = new_col
                    this_action_reward = self.rewards["p_es_hg"]
                    self.total_golds -= 1
                    self.agent.has_box = False
                else:
                    self.agent.row = new_row
                    self.agent.col = new_col
                    this_action_reward = self.rewards["n_es_hng"]
            else:
                print("Bug to fix: This condition is not predicted.")
                this_action_reward = 0
            
            if self.total_golds == 0:
                self.done = True
            else:
                self.done = False
            
            return this_action_reward, self.done

    def add_wall(self,wall_xy):
        
        for loc in wall_xy:
            if loc[0] < self.nrows and loc[1] < self.ncols:
                self.domain_mat[loc[0]][loc[1]].update("wall")
                self.n_wall += 1
   
    def add_agent(self,agent_loc):

        for loc in agent_loc:
            if ((loc[0] < self.nrows and loc[1] < self.ncols) and
                self.domain_mat[loc[0]][loc[1]].is_available()):

                self.agent = DAgent(loc)
                self.can_add_agent = False # limit to one agent for now. 

    def add_gold(self,gold_loc):
        for loc in gold_loc:
            if ((loc[0] < self.nrows and loc[1] < self.ncols) and
                self.domain_mat[loc[0]][loc[1]].is_available()):
                self.domain_mat[loc[0]][loc[1]].update("gold")
                self.total_golds += 1

    def add_storage(self,storage_loc):
        for loc in storage_loc:
            if ((loc[0] < self.nrows and loc[1] < self.ncols) and
                self.domain_mat[loc[0]][loc[1]].is_available()):
                self.domain_mat[loc[0]][loc[1]].update("storage")
                self.n_storage += 1
   
    def compute_color_tensor(self):
        """color tensor will be used to presetnation purposes."""
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.agent.row == i and self.agent.col == j:
                    r,g,b = self.domain_mat[i][j].get_color(self.agent)
                   
                else:
                    r,g,b = self.domain_mat[i][j].get_color(None)
                
                self.color_space_holder[0][i][j] = r/255
                self.color_space_holder[1][i][j] = g/255
                self.color_space_holder[2][i][j] = b/255
        
    def comput_code(self):
        """code matrix will be used as an input for linear neural network"""
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.agent.row == i and self.agent.col == j:                    
                     val = self.domain_mat[i][j].get_code(self.agent)                   
                else:
                     val = self.domain_mat[i][j].get_code(None)
                
                self.code_space_holder[i][j] =  val

    def get_state(self):
        """
        Returns cube codes in case of linear prediction (1D), or 3 channel 
        pytorch style (color channel, height, width) image tensor (2D).
        """

        if self.domain_type == "1D":
            self.comput_code()
            # use reshape (1,-1) if data is a single sample.
            return self.code_space_holder.reshape(1,-1)/10

        if self.domain_type == "2D":
            self.compute_color_tensor()
            # return np.copy(self.color_space_holder)
            
            fig = plt.figure()
            im = plt.imshow(self.plot_domain(False), interpolation=None)     
            # print(im)   
            plt.tight_layout(pad=0)
            fig = plt.gcf()
            fig.set_size_inches(1,1)
            ax = plt.gca()
            ax.set_xticks(np.arange(0.5, self.ncols, 1))
            ax.set_yticks(np.arange(0.5, self.nrows, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            image = fig2img(fig)
            image = image.convert('RGB')
            #--------------------------------------------
            # Uncomment to see an example of image quality
            # image.show()
            #--------------------------------------------
            np_image = np.asarray(image).transpose(2,0,1)
            plt.close(12905)
            return np_image

        return None
 
    def plot_domain(self,individual=True, into_folder=False):
        self.compute_color_tensor()
        canvas = np.transpose(np.copy(self.color_space_holder),(1,2,0))
        
        if individual:
            fig = plt.figure(1, figsize=(9, 6))
            gridspec.GridSpec(1,3)
                        
            # plotting domain
            plt.subplot2grid((1,3),(0,0), colspan=2, rowspan=1)
            ax = plt.gca()
     
            ax.set_xticks(np.arange(0.5, self.ncols, 1))
            ax.set_yticks(np.arange(0.5, self.nrows, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])            
            
            plt.grid(True)
            plt.imshow(canvas, interpolation='none') 
            
            # plotting parameters
            plt.subplot2grid((1,3),(0,2), colspan=1, rowspan=1)
            plt.axis('off')
            # plt.text(0,0.95, "Param1 goes here")
            # plt.text(0,0.85, "Param2 goes here")
            # plt.text(0,0.75, "Param3 goes here")
            
            if into_folder:                
                fig.savefig(os.path.join(self.domain_name+'.pdf'))
                return            
            
            plt.show()

        else:
            return canvas


    def step(self, action):
        """
        Takes the action and returns observation (new_state), reward, 
        termination. I chose function name following OpenAI convention.  
        
        Input:
            |action: 0,1,2,3 or "Up", "Down", "Left", "Right"
            
        """
        try:
            if action in self.actions:
                act = action
            elif action in range(4):
                act = self.actions[action]
            else:
                print("Action is not defined --> "+str(action))
        except Exception as e:
            print("Problem with action."+str(e))
            return

        this_action_reward, done = self.action(act)
        new_state = self.get_state()

        return new_state, this_action_reward, done

     
    def build_domain(self, location_list):
        """
        Having the different features location, this funtion builds the domain.
        Input:
            |location_list: [wall, agent, storage, gold]
        """

        self.initiate_domain()
        self.add_wall(location_list[0])
        self.add_agent(location_list[1])
        self.add_storage(location_list[2])
        self.add_gold(location_list[3])

        if not self.original_domain_location:
            self.original_domain_location = location_list
              
            
    def reset(self, new_start_loc=False):
        if not self.original_domain_location:
            return

        if new_start_loc:
            all_loc = ((i,j) for i in range(self.nrows) for j in range(self.ncols))
         
            possible_loc = itertools.chain.from_iterable(self.original_domain_location)
            possible_loc_tp = (tuple(i) for i in possible_loc)
            p_loc = set(all_loc) - set(possible_loc_tp)
            agent_new_loc = random.sample(p_loc,1)[0]
            dl = self.original_domain_location
            dl[1] = [[agent_new_loc[0],agent_new_loc[1]]]
            self.build_domain(dl)
            return
        
        self.build_domain(self.original_domain_location)

    def set_original_location(self, original_domain_location):
        self.original_domain_location = original_domain_location

    def save_domain(self, output_folder, domain_param):
        # self.domain_name = self.generate_domain_name(domain_param)
        with open(f'{output_folder}/{self.domain_name}.pkl', 'wb') as b:
            pickle.dump(self.original_domain_location,b)


    def generate_domain_name(self, domain_param=None):
        

        if not domain_param:
            domain_name = f"domain_{str(self.ncols)}_{str(self.nrows)}"\
             f"_w{str(self.n_wall)}_g{str(self.total_golds)}_s{str(self.n_storage)}_"\
             f"{self.domain_type}_{str(self.domain_number)}"
        else:
            num_wall = domain_param["num_wall"]
            num_gold = domain_param["num_gold"]
            num_storage = domain_param["num_storage"]
            self.domain_number = domain_param["domain_number"]
            domain_name = f"domain_{str(self.ncols)}_{str(self.nrows)}"\
             f"_w{str(num_wall)}_g{str(num_gold)}_s{str(num_storage)}_"\
             f"{self.domain_type}_{str(self.domain_number)}"
             

        self.domain_name = domain_name

    def load_domain(self, output_folder, domain_params):
   
        self.generate_domain_name(domain_params)

        try:
            with open(f'{output_folder}/{self.domain_name}.pkl', 'rb') as b:
                ol = pickle.load(b)
            self.build_domain(ol)
            return True
        except:
            return False

    def n_actions(self):
        return len(self.actions)

    
def fig2img(fig):

    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=64)
    # fig.savefig('figure.png', format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    return img.convert('RGB')
    


if __name__=="__main__":
    import random
    import matplotlib.animation as animation
    mydomain = Domain((10,10), "2D", 123)
    mywall = []
    mystorage = []
    mygold=[]

    mywall.extend([[i,3] for i in range(0,3)])
    mywall.extend([[i,7] for i in range(2,6)])
    mywall.extend([[8,j] for j in range(2,4)])

    mygold.extend([[4,5],[8,6], [4,9]])

    mystorage.extend([[0,4],[0,5],[1,4],[1,5]])

    myagent = [[1,1]]

    mydomain.build_domain([mywall, myagent, mystorage, mygold])
    mydomain.generate_domain_name()
    mydomain.plot_domain(into_folder=True)
    # print(mydomain.get_state().shape)
    a = mydomain.get_state()
    print(a.shape)

    # mydomain.reset(new_start_loc=True)

    mydomain.get_state()




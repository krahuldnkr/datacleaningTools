"""finds the value function using Bellman equation for the given policy."""
import numpy as np
import matplotlib.pyplot as plt

class mdp_gridworld:

    def __init__(self, height, width, start, terminal_state, threshold, discount_factor):
        """_summary_

        Args:
            height (int): table height
            width (int): table width
            start (tuple): MDP start state description
        """
        self.state_table = np.random.randn(height,width)
        self.state = start
        self.threshold = threshold
        self.gamma = discount_factor

        # WHEN THERE is a terminal state
        if(terminal_state != (-1, -1)):
            self.state_table[terminal_state[0]][terminal_state[1]] = 0

        # check if start is bounded by the height and width of the initialised table
        if((start[0] >= height) or (start[1] >= width) or (start[0] < 0) or (start[1] < 0)):
            raise Exception("starting state is out of bounds") 


    def reward_defn(self, action):
        """generates reward based on action taken from the current state
        Args:
            state (tuple): description of the state
            action (tuple): description of the action

        Returns:
            int: reward scalar value
        """
        (height, width) = self.state_table.shape

        # if input states is A or B then return the reward 
        # independent of the action taken
        if(self.state[0] == 0 and self.state[1] == 1):
            return 10
        elif(self.state[0] == 0 and self.state[1] == 3):
            return 5
        else:
            # if state lies on the edge of the table,
            # return -1 if the action taken takes you out of the grid 
            if (self.state[0] == 0 and action[0] == -1):
                return -1
            elif (self.state[0] == height-1 and action[0] == 1):
                return -1
            elif(self.state[1] == 0 and action[1] == -1):
                return -1
            elif(self.state[1] == width-1 and action[1] == 1):
                return -1

        return 0


    def state_transition(self, action):
        """Given action, this method updates the state of the agent.

        Args:
            action (tuple): applied action by the agent.
        """
        (height, width) = self.state_table.shape
        new_state = self.state
        # if input states is A or B then return the reward 
        # independent of the action taken
        if(self.state[0] == 0 and self.state[1] == 1):
            new_state = (height-1, self.state[1])

        elif (self.state[0] == 0 and self.state[1] == 3):
            new_state = (2, self.state[1])

        else:
            new_state_height = self.state[0] + action[0]
            new_state_width = self.state[1] + action[1]

            #print(new_state_height, new_state_width)
            # if state lies on the edge of the table,
            # return -1 if the action taken takes you out of the grid 
            if (new_state_height >=0 and new_state_height < height and new_state_width >=0 and new_state_width < width):
                new_state = (new_state_height, new_state_width)

        return new_state
    
    def get_state_table(self):
        print(self.state_table)
        return self.state_table
    
    def iterative_policy_evaluation(self):
        # used for appproximating V 
        del_step = np.Inf; 
        print(del_step)
        self.get_state_table() 

        iter = 0
        del_step_array = []
        while (del_step > self.threshold):
            print('inside while')
            del_step = 0
            iter =iter+1


            for i in range(self.state_table.shape[0]):
                for j in range(self.state_table.shape[1]):
                    
                    current_state_value = self.state_table[i][j]
                    updated_state_value = 0
                    self.state = (i, j)

                    # policy is equiprobable, with U=(-1, 0), D = (1, 0), L = (0, -1), R = (0 ,1)
                    action = (-1, 0)    # UP
                    next_state_up = self.state_transition(action)
                    updated_state_value = self.reward_defn(action) + self.gamma * self.state_table[next_state_up[0]][next_state_up[1]]

                    print("after up",updated_state_value)
                    action = (1, 0) # DOWN
                    next_state_dwn = self.state_transition(action)
                    updated_state_value = updated_state_value + self.reward_defn(action) + self.gamma * self.state_table[next_state_dwn[0]][next_state_dwn[1]]

                    print("after dwn",updated_state_value)
                    action = (0, -1) # Left
                    next_state_lft = self.state_transition(action)
                    updated_state_value = updated_state_value + self.reward_defn(action) + self.gamma * self.state_table[next_state_lft[0]][next_state_lft[1]]

                    print("after lft",updated_state_value)
                    action = (0, 1) # Right
                    next_state_rt = self.state_transition(action)
                    updated_state_value = updated_state_value + self.reward_defn(action) + self.gamma * self.state_table[next_state_rt[0]][next_state_rt[1]]

                    print("after rt",updated_state_value)
                    # updating the value of the current state
                    self.state_table[i][j] = updated_state_value

                    # max change in the values of states in this iteration.
                    del_step = np.max([del_step, np.abs(current_state_value - updated_state_value)])
                    
            del_step_array.append(del_step)  
            if((del_step)>100):
                print("iter", iter)
                break


                
                
            print("convergence stats: ")
            print(del_step)        
        print("update table after policy evaluation: ")
        self.get_state_table()  
        return del_step_array
        plt.plot
    def plotList(self, list_data):
        data = np.array(list_data)
        figure = plt.figure(figsize=(20, 20))
        ax = figure.add_subplot(1, 1, 1, xticks =[], yticks=[])
        ax.set_title('Convergence Data for Del Step')
        ax.plot(data)

        plt.show()
# ?????????????? observation:: why for a discount factor of >0.2, the values in the state are going to infinity.
gridWrld_obj = mdp_gridworld(height=5, width=5, start=(4, 4), terminal_state=(-1, -1), threshold=0.0001, discount_factor=0.2)
convergence_data = gridWrld_obj.iterative_policy_evaluation()
gridWrld_obj.plotList(convergence_data)
# Let us unit test the state transition block

#action = (0,1)
#print("next state: ",gridWrld_obj.state_transition(action))
#print("next reward: ", gridWrld_obj.reward_defn(action))

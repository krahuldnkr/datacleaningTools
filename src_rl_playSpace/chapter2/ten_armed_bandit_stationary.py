# A simple bandit algorithm
import numpy as np
import matplotlib.pyplot as plt

class multi_armed_bandit_stationary:

    def __init__(self, num_actions, mean_rewards, q_init, step_size, eps_greedy, var_rewards, ucb_exp_degree):
        """_summary_

        Args:
            num_actions (_type_): _description_
            q_init (_type_): _description_
            step_size (_type_): _description_
            eps_greedy (_type_): _description_
        """
        self.N = list(0 for i in range(num_actions))
        self.Q = list(q_init for i in range(num_actions))
        self.step_size = step_size
        self.eps_greedy = eps_greedy
        self.mean_rewards = mean_rewards
        self.track_action = list()
        self.var_rewards = var_rewards
        self.ucb_degree = ucb_exp_degree

    def bandit_algo(self, numLoop):
        """_summary_
        """
        for _ in range(numLoop):
            act = self._choose_action()
            #act = self._ucb_action_selection(numLoop)
            reward = self._reward_algo(act)
            self.N[act] += 1
            #self.Q[act] = self.Q[act] + (1/self.N[act]) * (reward - self.Q[act])
            self.Q[act] = self.Q[act] + (self.step_size) * (reward - self.Q[act])
            disp_processing = int(self.Q[act]*100)

            if(act==2 or act==4):
                self.track_action.append((act, disp_processing/100))
        return self.Q, self.N, self.track_action

    def _choose_action(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        toss_coin = np.random.randn(1,1)
        # find the action with maximum value currently
        
        best_action = self.Q.index(max(self.Q))

        choosen_action = best_action
        if(toss_coin > self.eps_greedy):
            random_action = np.random.randint(0, len(self.Q)-1)
            if(random_action >= best_action):
                random_action += 1
            choosen_action = random_action

        return choosen_action
    
    def _reward_algo(self, action):
        """given  action, return reward

        Args:
            action (_type_): _description_
        """
        reward = np.random.normal(self.mean_rewards[action], self.var_rewards)
      
        return reward
    
    def _ucb_action_selection(self, numLoop):
        """_summary_
        """
        # degree of exploration
        action_estimate_uncertainty = list(self.Q[i] + self.ucb_degree*np.sqrt(np.log(numLoop)/self.N[i]) for i in range(len(self.Q)))
        best_action = action_estimate_uncertainty.index(max(action_estimate_uncertainty))

        return best_action

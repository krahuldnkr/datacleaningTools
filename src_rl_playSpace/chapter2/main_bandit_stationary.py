from ten_armed_bandit_stationary import multi_armed_bandit_stationary

num_actions = 10
mean_rewards = [0.2, -0.8, 1.7, 0.4, 1.2, -1.5, -0.1, -1.1, 0.8, -0.78]
q_init = 5
step_size = 0.1
eps_greedy = 1 
var_rewards = 1 # if variance is zero, correct values are easily estimated and no chance of getting suboptimal solutions.
ucb_exp_deg = 2

banditObj = multi_armed_bandit_stationary(num_actions, mean_rewards, q_init, step_size, eps_greedy, var_rewards, ucb_exp_degree=ucb_exp_deg)
q_values, action_taken, track_action = banditObj.bandit_algo(1000)
#!!!!!!!  [43, 28, 113, 22, 696, 16, 32, 10, 19, 21]  <-- found with epsilon of 0.1 (suboptimal solution)
#!!!!!!!  [53, 31, 247, 26, 520, 32, 20, 30, 24, 17]  <-- found with epsilon of 0.2 (suboptimal solution)
# after many runs, one result was [66, 15, 45, 16, 756, 17, 17, 22, 23, 23] <-- found with epsilon of 0.02, 
# corresponding q_values: [0.2093723702708682, -0.9648237563652473, 1.442477858340854, 0.397319000188266, 1.2023401854044733, -1.5123096990383988, -0.5125415212610334, -1.1567131868076221, 0.7621189625374282, -0.6775814796925945]
# this above observation could be attributed to the randomness.

# given good exploration opportunity to determine good estimates for all the actions. we can always avoid the above issue. this is observed 
# that after applying ucb based action selection, the above issue got resolved with fewer step size.
# As discussed in the text, ucb is difficult to be extended for non-stationary case. so next lets delve into a non-stationary problem.  

# TODO:: Need to check for how to apply and tune exploration techniques aptly.
print("q_vals: ",q_values)
print("action_nums: ",action_taken)
print(sum(q_values)-sum(mean_rewards))
print("action seq: ", track_action[1:500])

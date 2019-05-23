# 1.) We have d arms. For example, arms have ads that we display to users each time they
# connect to a webpage.
#
# 2.) Each time a user connects to this webpage, that makes a round
#
# 3.) At each round n, we choose one ad to display to the user.
#
# 4.) At each roun n, ad i gives reward ri(n) E {0,1}: ri(n) = 1 if the user clicked
# on the ad i , 0 if the user didn't
#
# 5.) Our goal is to maximize the total reward we get over many rounds.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reinforcement Learning
# Import the dataset
# If the user clicks on the ad it gets a 1, else return 0
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Implementing the UCB Algorithm
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualizing the Results
plt.hist('ads_selected')
plt.title('Histogram of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

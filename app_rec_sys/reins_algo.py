import sys, os
import numpy as np 
import pandas as pd

# Epsilon greedy algorithm 
def epsilon_greedy(dataset_path:str)-> pd.DataFrame:

    df = pd.read_csv(dataset_path)

    observation_space = df["user_id"]
    action_space = df["recipe_id"]
    reward = df["rating"]

    eps = 0.2
    n_prod = 10000
    n_ads = len(observation_space)
    Q = np.zeros(n_ads)
    N = np.zeros(n_ads)
    total_reward = 0
    avg_rewards = []

    # plotting using dataframe
    df_reward_comparison = pd.DataFrame(avg_rewards, columns=['A/B/n'])

    # Main Algorithm
    ad_chosen = np.random.randint(n_ads)
    for i in range(n_prod):
        R = df.iloc[ad_chosen]["rating"]
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)
        if np.random.uniform() <= eps:
            ad_chosen = np.random.randint(n_ads)
        else:
            ad_chosen = np.argmax(Q)
    df_reward_comparison['e-greedy: {}'.format(eps)] = avg_rewards

    # Dataframe with N and Q values
    final_selection_df = pd.DataFrame(data=N, columns=["N_VALUE"])
    final_selection_df["Q_VALUE"] = Q

    # Takes good rewards
    good_rewards = final_selection_df[(final_selection_df["N_VALUE"] > 0) & (final_selection_df["Q_VALUE"]==5)]
    food_index = good_rewards.sort_values(by=["N_VALUE"], ascending=False).index[:20]

    located_food_index_df = df.iloc[food_index]

    return located_food_index_df


# UCB algorithm
def ucb(dataset_path: str)-> pd.DataFrame:
    df = pd.read_csv(dataset_path)


    observation_space = df["user_id"]
    action_space = df["recipe_id"]
    reward = df["rating"]


    c = 1
    n_prod = 10000
    n_ads = len(observation_space)
    ad_indices = np.array(range(n_ads))
    Q = np.zeros(n_ads)
    N = np.zeros(n_ads)
    total_reward = 0
    avg_rewards = []

    # Main algorithm
    for t in range(1, n_prod + 1):
        if any(N==0):
            ad_chosen = np.random.choice(ad_indices[N==0])
        else:
            uncertainty = np.sqrt(np.log(t) / N)
            ad_chosen = np.argmax(Q + c * uncertainty)
        R = df.iloc[ad_chosen]["rating"]
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        total_reward += R
        avg_reward_so_far = total_reward / t
        avg_rewards.append(avg_reward_so_far)

    # Dataframe with N and Q values
    final_selection_df_ucb = pd.DataFrame(data=N, columns=["N_VALUE"])
    final_selection_df_ucb["Q_VALUE"] = Q

    # Takes good rewards
    good_rewards_ucb = final_selection_df_ucb[(final_selection_df_ucb["N_VALUE"] > 0) & (final_selection_df_ucb["Q_VALUE"]==5)]
    food_index_ucb = good_rewards_ucb.sort_values(by=["N_VALUE"], ascending=False).index[:20]

    located_ucb = df.iloc[food_index_ucb]
    
    return located_ucb

# dataset_relative_path = "data/interactions_train.csv"
# predictions = epsilon_greedy(dataset_path = dataset_relative_path)
# print(predictions)

# predictions_ucb = ucb(dataset_path= dataset_relative_path)
# print(predictions_ucb)


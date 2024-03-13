import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from LQR_CBF_rrtStar import LQRrrtStar
from tracking.cbf_qp_tracking import UnicyclePathFollower
from utils import env, plotting

import gc
import stopit
from multiprocessing import Process, Queue

def planning_wrapper(lqr_rrt_star, result_queue):
    waypoints = lqr_rrt_star.planning()
    result_queue.put(waypoints)  # Use a queue to safely return data from the process

@stopit.threading_timeoutable(default=0)
def planning(x_start, x_goal, visibility, path_saved):
    if visibility:
        iter_max = 3000
        iter_max = 4000
    else:
        iter_max = 1000
        iter_max = 2000
    lqr_rrt_star = LQRrrtStar(x_start=x_start, x_goal=x_goal,
                              max_sampled_node_dist=1.0,
                              max_rewiring_node_dist=2,
                              goal_sample_rate=0.1,
                              rewiring_radius=2,  
                              iter_max=iter_max,
                              solve_QP=False,
                              visibility=visibility,
                              show_animation=False,
                              path_saved=path_saved)
    t1 = time.time()
    waypoints = lqr_rrt_star.planning()
    t2 = time.time()
    time_took = t2-t1

    del lqr_rrt_star
    gc.collect()

    if waypoints is None:
        return 0
    return time_took

@stopit.threading_timeoutable(default=[-1, -1])
def following(path_saved):
    try:
        waypoints = np.load(path_saved)
    except:
        return -1
    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting(x_init, x_goal)
    env_handler = env.Env()

    print("Waypoints information, length: ", len(waypoints), waypoints[-2])
    path_follower = UnicyclePathFollower('unicycle2d', x_init, waypoints,
                                         alpha=2.0,
                                         show_animation=False,
                                         plotting=plot_handler,
                                         env=env_handler)
    unexpected_beh, early_violation = path_follower.run(save_animation=False)

    del path_follower
    gc.collect()

    return unexpected_beh, early_violation

def evaluate(num_runs=10):
    # set the directory name for this evaluation, the name include the date and time
    directory_name = time.strftime("%y%m%d-%H%M")
    # make a directory if it is empty
    os.makedirs(f'output/{directory_name}')
    # create a csv file
    with open(f'output/{directory_name}/evaluated.csv', 'w') as f:
        f.write("Visibility,Time,Unexpected_beh,Early_Violation\n")
    for visibility in [False, True]:
        for i in range(num_runs):
            print(f"\nVisibility: {visibility}, Run: {i+1}")

            x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
            x_goal = (10.0, 2.0)  # Goal node
            # type = 1 (large env)
            x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
            x_goal = (25.0, 3.0)  # Goal node
            
            if visibility:
                path_saved = os.getcwd()+f"/output/{directory_name}/state_traj_vis_{i+1:03d}.npy"
            else:
                path_saved = os.getcwd()+f"/output/{directory_name}/state_traj_ori_{i+1:03d}.npy"

            # loop until the path is generated
            while True:
                time_took = planning(x_start, x_goal, visibility, path_saved, timeout=50)
                # fonud a path
                if time_took != 0:
                    break
            unexpected_beh, early_violation = following(path_saved, timeout=50)
            print(f"Unexpected_beh: {unexpected_beh}, Early Violation: {early_violation}, Time: {time_took}\n")
            # save the results with csv
            with open(f'output/{directory_name}/evaluated.csv', 'a') as f:
                f.write(f"{int(visibility)},{time_took},{unexpected_beh},{early_violation}\n")

    return f'output/{directory_name}/'

def following_only(csv_path):
    with open(f'{csv_path}/re-evaluated.csv', 'w') as f:
        f.write("Visibility,Time,Unexpected_beh,Early_Violation\n")
    for visibility in [True, False]:
        visibility_df = pd.read_csv(f'{csv_path}/evaluated.csv')
        visibility_df = visibility_df[visibility_df['Visibility'] == int(visibility)]
        visibility_times = visibility_df['Time'].values.tolist()
        
        for i, time_val in enumerate(visibility_times):
            print(f"\nVisibility: {visibility}, Run: {i+1}")

            if visibility:
                path_saved = os.getcwd()+f"/{csv_path}/state_traj_vis_{i+1:03d}.npy"
            else:
                path_saved = os.getcwd()+f"/{csv_path}/state_traj_ori_{i+1:03d}.npy"

            unexpected_beh, early_violation = following(path_saved)
            if unexpected_beh == -1:
                continue
            print(f"Unexpected_beh: {unexpected_beh}, Early Violation: {early_violation}\n")
            with open(f'{csv_path}/re-evaluated.csv', 'a') as f:
                f.write(f"{int(visibility)},{time_val},{unexpected_beh},{early_violation}\n")
            

def plot(csv_path, csv_name="evaluated.csv"):
    plt.clf()
    plt.close()  # Close the first figure

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path+csv_name, dtype={'Visibility': int, 'Time': float, 'Unexpected_beh': int, 'Early_Violation': int})

    # Preprocess: Exclude trials where path was not generated (unexpected behavior is -1)
    df_filtered = df[df['Unexpected_beh'] != -1]

    # Classify the results into 'Fail' or 'Success'
    df_filtered['Result'] = df_filtered['Unexpected_beh'].apply(lambda x: 'Fail' if x > 0 else 'Success')

    # Summarize results based on visibility
    success_summary = df_filtered[df_filtered['Result'] == 'Success'].groupby('Visibility').size()
    violation_summary = df_filtered[df_filtered['Early_Violation'] > 0].groupby('Visibility').size()

    # Create a subplot with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    # Plot the success rate in the first subplot
    success_summary.plot(kind='bar', ax=ax1)
    ax1.set_title('Success Rate by Visibility')
    ax1.set_xlabel('Visibility')
    ax1.set_ylabel('Number of Trials')
    ax1.set_xticklabels(['False', 'True'], rotation=0)

    # Add labels on top of each bar
    for i, v in enumerate(success_summary.values):
        ax1.text(i, v, str(v), ha='center', va='bottom')

    # Plot the early violation rate in the second subplot
    violation_summary.plot(kind='bar', ax=ax2)
    ax2.set_title('Early Violation Rate by Visibility')
    ax2.set_xlabel('Visibility')
    ax2.set_ylabel('Number of Trials')
    ax2.set_xticklabels(['False', 'True'], rotation=0)

    # Add labels on top of each bar
    for i, v in enumerate(violation_summary.values):
        ax2.text(i, v, str(v), ha='center', va='bottom')

    # Plot the mean, variance, third quantile, and first quantile of time using a violin plot in the second subplot
    ax3.violinplot(dataset=[df_filtered[df_filtered['Visibility'] == False]['Time'], df_filtered[df_filtered['Visibility'] == True]['Time']],
                   positions=[0, 1], showmeans=True)
    ax3.set_title('Mean, Variance, Third Quantile, and First Quantile of Time by Visibility')
    ax3.set_xlabel('Visibility')
    ax3.set_ylabel('Time')
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['False', 'True'], rotation=0)

    plt.tight_layout()
    plt.savefig(csv_path+"evaluate.PNG")
    plt.show()  # Plot the second figure

    # Return DataFrame for further inspection if needed
    #summary.reset_index()

if __name__ == "__main__":
    csv_path = evaluate(num_runs=100)
    plot(csv_path)
    # plot("", "type2.csv")
    # plt.close()

    # csv_path = "output/240225-0430"
    # following_only(csv_path)
    # plot("output/240225-0430/", "re-evaluated.csv")
    # plt.close()
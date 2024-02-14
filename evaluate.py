import math
import numpy as np
import os
import time
import pandas as pd

from LQR_CBF_rrtStar import LQRrrtStar
from tracking.cbf_qp_tracking import UnicyclePathFollower



def run(x_start, x_goal, visibility, path_saved):
    lqr_rrt_star = LQRrrtStar(x_start=x_start, x_goal=x_goal,
                              max_sampled_node_dist=1.0,
                              max_rewiring_node_dist=2,
                              goal_sample_rate=0.1,
                              rewiring_radius=2,  
                              iter_max=1000,
                              solve_QP=False,
                              visibility=visibility,
                              show_animation=False,
                              path_saved=path_saved)
    t1 = time.time()
    waypoints = lqr_rrt_star.planning()
    t2 = time.time()
    time_took = t2-t1

    if waypoints is None:
        return time_took, -1
    x_init = waypoints[0]
    obs = np.array([0.5, 0.3, 0.1]).reshape(-1, 1) #FIXME: effectless in this case
    path_follower = UnicyclePathFollower('unicycle2d', obs, x_init, waypoints,
                                         alpha=2.0,
                                         show_obstacles=False,
                                         show_animation=False)
    unexpected_beh = path_follower.run(save_animation=False)

    return time_took, unexpected_beh

def evaluate():
    # set the directory name for this evaluation, the name include the date and time
    directory_name = time.strftime("%Y%m%d-%H%M%S")
    # make a directory if it is empty
    os.makedirs(f'output/{directory_name}')
    # create a csv file
    with open(f'output/{directory_name}/evaluated.csv', 'w') as f:
        f.write("Visibility,Time,Unexpected_beh\n")
    NUM_RUNS = 10
    for visibility in [True, False]:
        for i in range(NUM_RUNS):
            print(f"\nVisibility: {visibility}, Run: {i+1}")
            x_start = (5.0, 5.0, math.pi/2)  # Starting node (x, y, yaw)
            x_goal = (10.0, 3.0)  # Goal node
            if visibility:
                path_saved = os.getcwd()+f"/output//{directory_name}/state_traj_vis_{i:03d}.npy"
            else:
                path_saved = os.getcwd()+f"/output//{directory_name}/state_traj_ori_{i:03d}.npy"
            time_took, unexpected_beh = run(x_start, x_goal, visibility, path_saved)
            print(f"Unexpected_beh: {unexpected_beh}, Time: {time_took}\n")
            # save the results with csv
            with open(f'output//{directory_name}/evaluated.csv', 'a') as f:
                f.write(f"{int(visibility)},{time_took},{unexpected_beh}\n")

    return f'output/{directory_name}/evaluated.csv'

def plot(csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path, dtype={'Visibility': int, 'Time': float, 'Unexpected_beh': int})

    # Preprocess: Exclude trials where path was not generated (unexpected behavior is -1)
    df_filtered = df[df['Unexpected_beh'] != -1]

    # Classify the results into 'Fail' or 'Success'
    df_filtered['Result'] = df_filtered['Unexpected_beh'].apply(lambda x: 'Fail' if x > 0 else 'Success')

    # Summarize results based on visibility
    summary = df_filtered.groupby(['Visibility', 'Result']).size().unstack(fill_value=0)

    # Prepare data for plotting
    time_success = df_filtered[df_filtered['Result'] == 'Success']['Time']
    time_fail = df_filtered[df_filtered['Result'] == 'Fail']['Time']

    # Visualize the results and time taken
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot 1: Time distribution for Success and Fail
    ax[0].hist([time_success, time_fail], color=['green', 'red'], label=['Success', 'Fail'], bins=20, alpha=0.7)
    ax[0].set_title('Time Distribution for Success and Fail Trials')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Number of Trials')
    ax[0].legend()

    # Plot 2: Summary table as a bar chart
    summary.plot(kind='bar', ax=ax[1])
    ax[1].set_title('Trial Results by Visibility')
    ax[1].set_xlabel('Visibility')
    ax[1].set_ylabel('Number of Trials')
    ax[1].set_xticklabels(['False', 'True'], rotation=0)

    plt.tight_layout()
    plt.show()

    # Return DataFrame for further inspection if needed
    summary.reset_index()

if __name__ == "__main__":
    csv_path = evaluate()
    #plot(f'output/20240214-101358/evaluated.csv')
    plot(csv_path)
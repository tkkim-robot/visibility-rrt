import math
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from LQR_CBF_rrtStar import LQRrrtStar
from tracking.cbf_qp_tracking import UnicyclePathFollower

import gc
from multiprocessing import Process, Queue

def planning_wrapper(lqr_rrt_star, result_queue):
    waypoints = lqr_rrt_star.planning()
    result_queue.put(waypoints)  # Use a queue to safely return data from the process

def run(x_start, x_goal, visibility, path_saved):
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
    # Use a Queue to receive the waypoints from the process
    result_queue = Queue()
    planning_process = Process(target=planning_wrapper, args=(lqr_rrt_star, result_queue))
    t1 = time.time()
    planning_process.start()
    planning_process.join(timeout=50)

    if planning_process.is_alive():
        # If the process is still alive after the timeout, terminate it
        planning_process.terminate()
        planning_process.join()  # Make sure the process has terminated
        print("Planning process was terminated due to timeout.")
        waypoints = None
    else:
        waypoints = result_queue.get() if not result_queue.empty() else None
    t2 = time.time()
    time_took = t2-t1

    if waypoints is None:
        return time_took, -1
    x_init = waypoints[0]
    obs = np.array([0.5, 0.3, 0.1]).reshape(-1, 1) #FIXME: effectless in this case
    print(len(waypoints), waypoints[-2])
    path_follower = UnicyclePathFollower('unicycle2d', obs, x_init, waypoints,
                                         alpha=2.0,
                                         show_obstacles=False,
                                         show_animation=False)
    unexpected_beh = path_follower.run(save_animation=False)

    del lqr_rrt_star
    del path_follower
    gc.collect()

    return time_took, unexpected_beh

def evaluate(num_runs=10):
    # set the directory name for this evaluation, the name include the date and time
    directory_name = time.strftime("%Y%m%d-%H%M%S")
    # make a directory if it is empty
    os.makedirs(f'output/{directory_name}')
    # create a csv file
    with open(f'output/{directory_name}/evaluated.csv', 'w') as f:
        f.write("Visibility,Time,Unexpected_beh\n")
    for visibility in [False, True]:
        for i in range(num_runs):
            print(f"\nVisibility: {visibility}, Run: {i+1}")

            x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
            x_goal = (10.0, 2.0)  # Goal node
            # type = 1 (large env)
            # x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
            # x_goal = (25.0, 3.0)  # Goal node
            
            if visibility:
                path_saved = os.getcwd()+f"/output//{directory_name}/state_traj_vis_{i+1:03d}.npy"
            else:
                path_saved = os.getcwd()+f"/output//{directory_name}/state_traj_ori_{i+1:03d}.npy"

            # loop until the path is generated
            while True:
                time_took, unexpected_beh = run(x_start, x_goal, visibility, path_saved)
                if unexpected_beh != -1:
                    break
            print(f"Unexpected_beh: {unexpected_beh}, Time: {time_took}\n")
            # save the results with csv
            with open(f'output//{directory_name}/evaluated.csv', 'a') as f:
                f.write(f"{int(visibility)},{time_took},{unexpected_beh}\n")

    return f'output/{directory_name}/'

def plot(csv_path):
    plt.clf()
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path+'evaluated.csv', dtype={'Visibility': int, 'Time': float, 'Unexpected_beh': int})

    # Preprocess: Exclude trials where path was not generated (unexpected behavior is -1)
    df_filtered = df[df['Unexpected_beh'] != -1]

    # Classify the results into 'Fail' or 'Success'
    df_filtered['Result'] = df_filtered['Unexpected_beh'].apply(lambda x: 'Fail' if x > 0 else 'Success')

    # Summarize results based on visibility
    summary = df_filtered.groupby(['Visibility', 'Result']).size().unstack(fill_value=0)


    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    summary.plot(kind='bar', ax=ax)
    ax.set_title('Trial Results by Visibility')
    ax.set_xlabel('Visibility')
    ax.set_ylabel('Number of Trials')
    ax.set_xticklabels(['False', 'True'], rotation=0)

    plt.tight_layout()
    plt.savefig(csv_path+"evaluate.PNG")
    plt.show()

    # Return DataFrame for further inspection if needed
    summary.reset_index()

if __name__ == "__main__":
    csv_path = evaluate(num_runs=2)
    #plot(f'output/20240214-101358/evaluated.csv')
    plot(csv_path)
    plt.close()
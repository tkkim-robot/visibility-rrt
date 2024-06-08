import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visibility_rrtStar import VisibilityRRTStar
from tracking.cbf_qp_tracking import UnicyclePathFollower
from utils import env, plotting

import gc
import stopit

env_type = env.type

def planning_wrapper(lqr_rrt_star, result_queue):
    waypoints = lqr_rrt_star.planning()
    result_queue.put(waypoints)  # Use a queue to safely return data from the process

@stopit.threading_timeoutable(default=[0,0])
def planning(x_start, x_goal, visibility, collision_cbf, path_saved):
    if env_type == 1:
        iter_max = 3000
    else:
        iter_max = 2000
    lqr_rrt_star = VisibilityRRTStar(x_start=x_start, x_goal=x_goal,
                                    max_sampled_node_dist=1.0,
                                    max_rewiring_node_dist=2,
                                    goal_sample_rate=0.1,
                                    rewiring_radius=2,  
                                    iter_max=iter_max,
                                    solve_QP=False,
                                    visibility=visibility,
                                    collision_cbf=collision_cbf,
                                    show_animation=False,
                                    path_saved=path_saved)

    waypoints, compute_node_time, rewire_time = lqr_rrt_star.planning()

    del lqr_rrt_star
    gc.collect()

    if waypoints is None:
        return 0, 0
    return compute_node_time, rewire_time


@stopit.threading_timeoutable(default=[-2, 0])
def following(path_saved, robot_type, test_type):
    try:
        waypoints = np.load(path_saved)
    except:
        return -1
    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting(x_init, x_goal)
    env_handler = env.Env()

    print("Waypoints information, length: ", len(waypoints), waypoints[-2])
    path_follower = UnicyclePathFollower(robot_type, x_init, waypoints,
                                         show_animation=False,
                                         plotting=plot_handler,
                                         env=env_handler)
    if test_type =='cbf_qp':
        if env_type == 1:
            unknown_obs = np.array([[13.0, 10.0, 0.5],
                                [12.0, 13.0, 0.5],
                                [15.0, 20.0, 0.5],
                                [20.5, 20.5, 0.5],
                                [24.0, 15.0, 0.5]])
        elif env_type == 2:
            unknown_obs = np.array([[9.0, 8.8, 0.3]]) # test for FOV 45, type 2 (small env)
        path_follower.set_unknown_obs(unknown_obs)
        print("set unknown obs")
    unexpected_beh, early_violation = path_follower.run(save_animation=False)

    del path_follower
    gc.collect()

    return unexpected_beh, early_violation

def evaluate(num_runs=10, robot_type='Unicycle2D', 
             algorithms=['lqr_rrt_star', 'lqr_cbf_rrt_star', 'visibility_rrt_star'],
             following_flag=True,
             following_robot_type='DynamicUnicycle2D',
             following_test_type='cbf_qp'):
    # set the directory name for this evaluation, the name include the date and time
    directory_name = time.strftime("%y%m%d-%H%M")
    # make a directory if it is empty
    os.makedirs(f'output/{directory_name}')
    # create a csv file
    with open(f'output/{directory_name}/evaluated.csv', 'w') as f:
        f.write("Algorithm,Time,Unexpected_beh,Early_Violation,NodeTime,RewireTime\n")
    for algorithm in algorithms:
        if algorithm == 'lqr_rrt_star':
            visibility_cbf = False
            collision_cbf = False
            abbrev = 'lqr'
        elif algorithm == 'lqr_cbf_rrt_star':
            visibility_cbf = False
            collision_cbf = True
            abbrev = 'cbf'
        elif algorithm == 'visibility_rrt_star':
            visibility_cbf = True
            collision_cbf = True
            abbrev = 'vis'
        for i in range(num_runs):
            print(f"\nAlgorithm: {algorithm}, Run: {i+1}")

            if env_type == 1:
                x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
                x_goal = (25.0, 3.0)  # Goal node
            elif env_type == 2:
                x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
                x_goal = (10.0, 2.0)  # Goal node

            path_saved = os.getcwd()+f"/output/{directory_name}/state_traj_{abbrev}_{i+1:03d}.npy"

            # loop until the path is generated
            while True:
                compute_node_time, rewire_time = planning(x_start, x_goal, visibility_cbf, collision_cbf, path_saved, timeout=50)
                # fonud a path
                if compute_node_time != 0:
                    break

            if following_flag:
                unexpected_beh, early_violation = following(path_saved, following_robot_type, following_test_type, timeout=50)
                print(f"Unexpected_beh: {unexpected_beh}, Early Violation: {early_violation}, NodeTime: {compute_node_time}, RewireTime: {rewire_time}\n")
                # save the results with csv
            else:
                unexpected_beh = -100 # didn't test it
                early_violation = -100
                print(f"NodeTime: {compute_node_time}, RewireTime: {rewire_time}\n")
            with open(f'output/{directory_name}/evaluated.csv', 'a') as f:
                f.write(f"{algorithm},{compute_node_time+rewire_time},{unexpected_beh},{early_violation},{compute_node_time},{rewire_time}\n")


    return f'output/{directory_name}/'

def following_test(csv_path, robot_type, test_type, algorithms=['lqr_rrt_star', 'lqr_cbf_rrt_star', 'visibility_rrt_star']):
    with open(f"{csv_path}/re-evaluated_{robot_type}_{test_type}.csv", 'w') as f:
        f.write("Algorithm,Time,Unexpected_beh,Early_Violation\n")
    for algorithm in algorithms:
        if algorithm == 'lqr_rrt_star':
            abbrev = 'lqr'
        elif algorithm == 'lqr_cbf_rrt_star':
            abbrev = 'cbf'
        elif algorithm == 'visibility_rrt_star':
            abbrev = 'vis'

        algorithm_df = pd.read_csv(f'{csv_path}/evaluated.csv')
        algorithm_df = algorithm_df[algorithm_df['Algorithm'] == algorithm]
        algorithm_times = algorithm_df['Time'].values.tolist()
        
        for i, time_val in enumerate(algorithm_times):
            print(f"\nAlgorithm: {algorithm}, Run: {i+1}")
            path_saved = os.getcwd()+f"/{csv_path}/state_traj_{abbrev}_{i+1:03d}.npy"

            unexpected_beh, early_violation = following(path_saved, robot_type, test_type, timeout=10)
            if unexpected_beh == -2:
                print("Time out") # unknown obs are blocking the path, but no collision
            elif unexpected_beh == -1:
                print("QP failed") # QP failed and possibly collide with obs
            print(f"Unexpected_beh: {unexpected_beh}, Early Violation: {early_violation}\n")
            with open(f"{csv_path}/re-evaluated_{robot_type}_{test_type}.csv", 'a') as f:
                f.write(f"{algorithm},{time_val},{unexpected_beh},{early_violation}\n")

def filter_outlier(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def plot_test_gatekeeper(csv_path, csv_name="evaluated.csv"):
    plt.clf()
    plt.close()  # Close the first figure

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path+csv_name, dtype={'Algorithm': str, 'Time': float, 'Unexpected_beh': int, 'Early_Violation': int, 'NodeTime': float, 'RewireTime': float})

    # -2: time out, -1: QP failed
    # but in gatekeeper experiment,there is no -2 case (no added obs)
    # exclude trials where path was not generated (unexpected behavior is -1)
    df_filtered = df[df['Unexpected_beh'] >= 0]

    # if unexpected_beh > 0, it violates visibility constraints
    df_filtered['Result'] = df_filtered['Unexpected_beh'].apply(lambda x: 'Fail' if x > 0 else 'Success')

    # Summarize results based on visibility
    success_summary = df_filtered[df_filtered['Result'] == 'Success'].groupby('Algorithm').size()
    violation_summary = df_filtered[df_filtered['Early_Violation'] > 0].groupby('Algorithm').size()
    # Fill missing categories with 0
    success_summary = success_summary.reindex(['lqr_rrt_star', 'lqr_cbf_rrt_star', 'visibility_rrt_star'], fill_value=0)
    violation_summary = violation_summary.reindex(['lqr_rrt_star', 'lqr_cbf_rrt_star', 'visibility_rrt_star'], fill_value=0)

    # Create a subplot with two subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))

    # Plot the success rate in the first subplot
    success_summary.plot(kind='bar', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, horizontalalignment='right')
    ax1.set_title('Success Rate by Visibility')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Number of Trials')
    # rotate x-axis labels

    # Add labels on top of each bar
    for i, v in enumerate(success_summary.values):
        ax1.text(i, v, str(v), ha='center', va='bottom')

    # Plot the early violation rate in the second subplot
    violation_summary.plot(kind='bar', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, horizontalalignment='right')
    ax2.set_title('Early Violation Rate by Visibility')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Number of Trials')

    # Add labels on top of each bar
    for i, v in enumerate(violation_summary.values):
        ax2.text(i, v, str(v), ha='center', va='bottom')

    # drop the outlier in NodeTime and RewireTime

    # NodeTime violin plot
    # remove top 3% of the data in each algorithm group
    # Filter out the top 3% of data in NodeTime and RewireTime columns for each algorithm
    df = filter_outlier(df, 'NodeTime')
    df = filter_outlier(df, 'RewireTime')
    sns.violinplot(data=df, x='Algorithm', y='NodeTime', ax=ax3)
    ax3.set_title('NodeTime by Algorithm')
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('NodeTime (seconds)')

    # Adding mean value text on NodeTime plot
    node_time_means = df.groupby('Algorithm')['NodeTime'].mean()
    # rearragne the order of the algorithms 
    node_time_means = node_time_means.reindex(['lqr_rrt_star', 'lqr_cbf_rrt_star', 'visibility_rrt_star'])
    for i, algorithm in enumerate(node_time_means.index):
        mean_val = node_time_means[algorithm]
        print(algorithm, mean_val)
        ax3.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='center', fontsize=10, color='white', bbox=dict(facecolor='red', alpha=0.5))

    # RewireTime violin plot
    sns.violinplot(data=df, x='Algorithm', y='RewireTime', ax=ax4)
    ax4.set_title('RewireTime by Algorithm')
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('RewireTime (seconds)')

    # Adding mean value text on RewireTime plot
    rewire_time_means = df.groupby('Algorithm')['RewireTime'].mean()
    rewire_time_means = rewire_time_means.reindex(['lqr_rrt_star', 'lqr_cbf_rrt_star', 'visibility_rrt_star'])

    for i, algorithm in enumerate(rewire_time_means.index):
        mean_val = rewire_time_means[algorithm]
        print(algorithm, mean_val)
        ax4.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='center', fontsize=10, color='white', bbox=dict(facecolor='red', alpha=0.5))

    plt.tight_layout()
    plt.savefig(csv_path+"evaluate.PNG")
    plt.show()  # Plot the second figure

def plot_test_cbf_qp(csv_path, csv_name="evaluated.csv"):
    plt.clf()
    plt.close()  # Close the first figure

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path+csv_name, dtype={'Algorithm': str, 'Time': float, 'Unexpected_beh': int, 'Early_Violation': int})


    # -2: time out, -1: QP failed or collide
    df['Result'] = df['Unexpected_beh'].apply(lambda x: 'Fail' if x == -1 else 'Success')

    # Summarize results based on visibility
    failure_summary = df[df['Result'] == 'Fail'].groupby('Algorithm').size()

    # Fill missing categories with 0
    failure_summary = failure_summary.reindex(['lqr_rrt_star', 'lqr_cbf_rrt_star', 'visibility_rrt_star'], fill_value=0)

    # Create a subplot with two subplots
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the failure rate
    failure_summary.plot(kind='bar', ax=ax)
    ax.set_title('Failure Rate by Visibility')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Number of Trials')

    # Add labels on top of each bar
    for i, v in enumerate(failure_summary.values):
        ax.text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(csv_path + "qp_failure_rate.png")
    plt.show()

    # Return DataFrame for further inspection if needed
    #summary.reset_index()

if __name__ == "__main__":
    #csv_path = evaluate(num_runs=100, robot_type='Unicycle2D', algorithms=['lqr_rrt_star', 'lqr_cbf_rrt_star', 'visibility_rrt_star'], following_flag=False)
    #csv_path = evaluate(num_runs=50, robot_type='Unicycle2D', algorithms=['lqr_rrt_star'], following_flag=True)
    
    # following test
    robot_type = 'DynamicUnicycle2D'
    #test_type = 'cbf_qp'
    test_type = 'gatekeeper'

    if env_type == 1:
        csv_path = "output/240312-2128_large_env"
    elif env_type == 2:
        csv_path = "output/240225-0430"

    #following_test(csv_path, robot_type, test_type) # test with dynamic unicycle model

    if test_type == 'cbf_qp':
        #plot_test_cbf_qp(csv_path, f"/re-evaluated_{robot_type}_{test_type}.csv")
        plot_test_cbf_qp(csv_path, f"/re-evaluated_{robot_type}_{test_type}.csv")
    elif test_type == 'gatekeeper':
        plot_test_gatekeeper(csv_path, f"/re-evaluated_{robot_type}_{test_type}.csv") 
    
    plt.close()
    


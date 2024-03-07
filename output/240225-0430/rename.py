import os

directory = 'output/240225-0430'  # Replace with the actual directory path

# Get a list of all the existing filenames
existing_filenames = [filename for filename in os.listdir(directory) if filename.endswith('.npy')]

# Separate filenames into "state_traj_ori" and "state_traj_vis"
ori_filenames = [filename for filename in existing_filenames if 'state_traj_ori' in filename]
vis_filenames = [filename for filename in existing_filenames if 'state_traj_vis' in filename]

# Sort the filenames in ascending order
ori_filenames.sort()
vis_filenames.sort()

# Initialize the new number for "state_traj_ori"
ori_new_number = 1

# Iterate through the existing "state_traj_ori" filenames
for filename in ori_filenames:
    # Extract the number from the filename
    number = int(filename.split('_')[-1].split('.')[0])
    
    # Generate the new filename
    new_filename = filename.replace(str(number).zfill(3), str(ori_new_number).zfill(3))
    
    # Rename the file
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    
    # Increment the new number for "state_traj_ori"
    ori_new_number += 1

# Initialize the new number for "state_traj_vis"
vis_new_number = 1

# Iterate through the existing "state_traj_vis" filenames
for filename in vis_filenames:
    # Extract the number from the filename
    number = int(filename.split('_')[-1].split('.')[0])
    
    # Generate the new filename
    new_filename = filename.replace(str(number).zfill(3), str(vis_new_number).zfill(3))
    
    # Rename the file
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    
    # Increment the new number for "state_traj_vis"
    vis_new_number += 1

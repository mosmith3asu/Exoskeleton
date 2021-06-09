from Modules.Utils import DataAnalysis

###################################################
# START PARAMS - CHANGE THESE VARIABLES
###################################################

ffmpeg_path = 'C:\\ffmpeg\\ffmpeg-20200831-4a11a6f-win64-static\\bin\\ffmpeg.exe'
save_dir = "C:\\Users\\mason\\Desktop\\Thesis\\Patient_files" # where all saved files will be stored

# Calculated Joint Angles
plot_import = True      # Plot the imported & processed joint angles
save_import = True     # Save the imported & processed joint angles as csv
filtered_import = True  # Filter the imported & processed joint angles
filter_iterations = 7   # Degree of filtering (higher = smoother)

# Simulation
save_animation = False   # Save simulation as mp4
show_animation = False  # Whether to show animation after save
sim_speed = 4
sim_bitrate = 1800
window_range = 800  # How many indexes in the future in view

"""
Setting Data Parameters
    -UNCOMMENT/CHANGE THE BELOW LIST (data_param) TO SELECT AND MODIFY PROCESSED DATA
    -Recomended settings are already defined
    -Structure of data parameters:
        data_param = [path,init_indexs,filter_iter]
    
Variables:
    *path*:         path to raw IMU data [quaternion,quaternion]
                    *Need to modify to include the path on your local directory*
                        
    init_idexs:     list of initialization points where knee set to zero flexion
                    Some trials initialization point is not stable so it is skipped
                        
"""
if __name__ == "__main__":
    path = 0  # named int, do not change
    init_idxs = 1  # named int, do not change
    filter_iter = 2  # named int, do not change

    data_param = [
         #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal1.csv', [200, 1431, 2562, 3601]
         #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal2.csv',      [268,2966,5349,7535]
        'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal3_180.csv',  [429,3280,6838,9434]
         #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_MocapTrial.csv',  [3176],10

        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S1.csv', [624] #[624, 5390, 9453, 16523, 19910]
         #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S2.csv', [0]
         #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S3.csv', [11150, 18580, 40731]

        #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S1.csv', [0]
        #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S2.csv', [2210,10364,13051,20501]
         #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S3.csv', [560,2451]
    ]
    data_name = data_param[path].split("\\")[-1].split(".")[0]

    # Calculate Joint Angles from Quaternion
    Quat2JointAngles = DataAnalysis.Quat2JointAngles()
    Quat2JointAngles.plot_processed = plot_import
    Quat2JointAngles.save_processed = save_import
    KHS= Quat2JointAngles.quat2KneeHipShank(quat_csv_path=data_param[path],
                                                init_idxs=data_param[init_idxs],
                                                name_prefix=data_name
                                                )

    # Simulate Calculated Joint Angles
    Sim = DataAnalysis.GaitSimulation(KHS,ffmpeg_path =ffmpeg_path)
    Sim.show_animation = show_animation
    Sim.save_animation = save_animation
    Sim.index_speed = sim_speed
    Sim.bitrate = sim_bitrate
    Sim.index_start = data_param[init_idxs][0]
    Sim.window_range = window_range
    Sim.data_name = data_name
    Sim.run()
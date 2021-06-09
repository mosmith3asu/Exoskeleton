import numpy as np
from numpy import genfromtxt,savetxt
from pyquaternion import Quaternion
from math import degrees
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from math import sin,cos

class GaitSimulation:
    def __init__(self,KHS,ffmpeg_path,init_idxs = None):
        root_dir = 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files'

        #self.ffmpeg_path = 'C:\\ffmpeg\\ffmpeg-20200831-4a11a6f-win64-static\\bin\\ffmpeg.exe'
        self.ffmpeg_path = ffmpeg_path
        self.deg2rad = 3.14 / 180.0

        # Simulation Settings
        self.save_animation = False  # Save simulation as mp4
        self.show_animation = True  # Whether to show animation after save
        self.data_name = ""
        self.leg_thickness = 2
        self.leg_color = 'k'
        self.joint_thickness = 20
        self.joint_color = 'r'
        self.footsize = 0.25
        self.foot_dir = -1

        self.ref_len = 0.5
        self.ref_thickness = 0.5
        self.ref_color = 'grey'
        self.ref_offset = 0.4
        self.enable_hip_reference = True
        self.enable_shank_reference= True
        self.enable_knee_reference= True


        self.sim_size = 2.5  # How big leg simulation window is
        self.window_range = 1000  # How many indexes in the future to view
        self.annote_loc = 0.8 * self.sim_size
        self.init_color = 'r'

        self.index_start = 0
        self.index_speed = 1
        self.fps = 6  # assign to match sample rate (10ms = 6fps)
        self.bitrate = 1800


        # Unpack Data
        kneen = 0  # named int, do not change
        hipn = 1  # named int, do not change
        shankn = 2  # named int, do not change

        self.knee_angles = KHS[:, kneen].tolist()
        self.hip_angles = KHS[:, hipn].tolist()
        self.shank_angles = KHS[:, shankn].tolist()
        if init_idxs==None: self.init_idxs = [0]


    def initialize_settings(self):
        plt.rcParams['animation.ffmpeg_path'] = self.ffmpeg_path
        self.Writer = animation.writers['ffmpeg']
        self.writer = self.Writer(fps=self.fps, metadata=dict(artist='Me'), bitrate=self.bitrate)
        self.n_frames = int((len(self.knee_angles) - self.index_start) / self.index_speed)  # Number of times the animation updates

    def initialize_plots(self):
        # Create Figure and plots
        self.fig = plt.figure(figsize=plt.figaspect(0.5))
        self.mng = plt.get_current_fig_manager()
        self.mng.full_screen_toggle()

        self.ax = self.fig.add_subplot(2, 2, 1)
        self.ax_KA = self.fig.add_subplot(2, 2, 2)
        self.ax_HA = self.fig.add_subplot(2, 2, 4)
        self.ax_SA = self.fig.add_subplot(2, 2, 3)

        self.ax_KA.set_ylim((min(-2, min(self.knee_angles)), min(60, max(self.knee_angles))))
        self.ax_HA.set_ylim((min(-20, min(self.hip_angles)), min(60, max(self.hip_angles))))
        self.ax_SA.set_ylim((min(-20, min(self.shank_angles)), min(60, max(self.shank_angles))))

        # Set Titles
        self.ax.set_title("Leg Simulation")
        self.ax_KA.set_title("Knee Displacment vs Time")
        self.ax_HA.set_title("Hip Displacment vs Time")
        self.ax_SA.set_title("Shank Displacment vs Time")

         # Plot Joint Angles
        self.ax_KA.plot(self.knee_angles)
        self.ax_HA.plot(self.hip_angles)
        self.ax_SA.plot(self.shank_angles)

         # Plot initialization indexes (verticle red lines)

         #init_idxs = data_param[1]
        self.ax_KA.vlines(self.init_idxs, min(self.knee_angles), max(self.knee_angles), colors=self.init_color, linestyles=':', label="Init Pt")
        self.ax_HA.vlines(self.init_idxs, min(self.hip_angles), max(self.hip_angles), colors=self.init_color, linestyles=':')
        self.ax_SA.vlines(self.init_idxs, min(self.shank_angles), max(self.shank_angles), colors=self.init_color, linestyles=':')

         # Plot zero displacement line
        self.ax_KA.plot([0 for pt in range(len(self.knee_angles))], c='k', linewidth=1, label="Joint F/E")
        self.ax_HA.plot([0 for pt in range(len(self.hip_angles))], c='k', linewidth=1)
        self.ax_SA.plot([0 for pt in range(len(self.shank_angles))], c='k', linewidth=1)

        self.ax_KA.legend()

    def perpendicular_vector(self,v):
        ax = v[0]
        ay = v[1]
        vperp = [-ay,ax]
        return vperp

    def arc_patch(self,center, radius, theta1,theta2, ax=None, resolution=15, **kwargs):
        # make sure ax is not empty

        if ax is None:
            ax = plt.gca()
        # generate the points
        theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
        points = np.vstack((radius * np.cos(theta) + center[0],
                            radius * np.sin(theta) + center[1]))

        points = np.append(points,np.array([[center[0]],[center[1]]]),axis=1)
        # build the polygon and add it to the axes
        poly = mpatches.Polygon(points.T, closed=True, **kwargs)
        ax.add_patch(poly)
        return poly

    def update(self,i):
        global list_KA
        global init_ang

        # Get index references
        index = self.index_start + i * self.index_speed

        ############################################
        ## Simulation ##############################
        ############################################
        self.ax.clear()
        self.ax.set_title(f'Leg Simulation {self.data_name}')
        self.ax.text(self.annote_loc, self.annote_loc, f"Index={index}")
        self.ax.set_xlim(-self.sim_size / 2, self.sim_size / 2)
        self.ax.set_ylim(-self.sim_size, 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # Calculate Joint Locations

        hip = [0, 0]
        vknee = [sin(self.hip_angles[index] * self.deg2rad), cos(self.hip_angles[index] * self.deg2rad)]
        uvknee = (-1 * np.array(vknee) / np.linalg.norm(np.array(vknee))).tolist()
        knee = [uvknee[0] + hip[0], uvknee[1] + hip[1]]

        vankle = [sin((self.hip_angles[index] - self.knee_angles[index]) * self.deg2rad),
                  cos((self.hip_angles[index] - self.knee_angles[index]) * self.deg2rad)]
        uvankle = (-1 * np.array(vankle) / np.linalg.norm(np.array(vankle))).tolist()
        ankle = [uvankle[0] + uvknee[0], uvankle[1] + uvknee[1]]

        vfoot = self.perpendicular_vector(uvankle)
        vfoot = (self.foot_dir * self.footsize * np.array(vfoot) / np.linalg.norm(np.array(vfoot))).tolist()
        foot = [ankle[0] + vfoot[0], ankle[1] + vfoot[1]]

        # Plot Leg in simulation
        self.ax.scatter([hip[0], knee[0], ankle[0]], [hip[1], knee[1], ankle[1]], s=self.joint_thickness, c=self.joint_color)
        self.ax.plot([hip[0], knee[0]], [hip[1], knee[1]], linewidth=self.leg_thickness, c=self.leg_color)
        self.ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]], linewidth=self.leg_thickness, c=self.leg_color)
        self.ax.plot([ankle[0], foot[0]], [ankle[1], foot[1]], linewidth=self.leg_thickness, c=self.leg_color)

        fill = True
        # Reference Lines
        if self.enable_hip_reference:
            ref_color = 'g'
            self.arc_patch((hip[0],hip[1]), self.ref_len, 270-self.hip_angles[index],270, ax=self.ax,fill=fill, color=ref_color )
            self.ax.text(hip[0] - self.ref_offset, hip[1], f'H={int(self.hip_angles[index])}',c=ref_color)

        if self.enable_knee_reference:
            ref_color = 'b'
            self.arc_patch((knee[0], knee[1]), self.ref_len, 270-self.hip_angles[index], 270-self.shank_angles[index], ax=self.ax, fill=fill,color=ref_color )
            self.ax.text(knee[0] - self.ref_offset, knee[1]- self.ref_len, f'K={int(self.knee_angles[index])}',c=ref_color )

        if self.enable_shank_reference:
            ref_len = self.ref_len
            ref_color ='r'
            if self.enable_knee_reference:
                ref_len = self.ref_len*0.5
            self.arc_patch((knee[0], knee[1]), ref_len, 270 - self.shank_angles[index],270, ax=self.ax,fill=fill, color=ref_color )
            self.ax.text(knee[0] - self.ref_offset, knee[1]-ref_len, f'S={int(self.shank_angles[index])}',c=ref_color)
        # Change Viewing Window
        self.ax_KA.set_xlim(index, index + self.window_range)
        self.ax_HA.set_xlim(index, index + self.window_range)
        self.ax_SA.set_xlim(index, index + self.window_range)

    def run(self):
        self.initialize_settings()
        self.initialize_plots()

        self.anim = animation.FuncAnimation(self.fig, self.update, interval=int(60/self.fps), frames=self.n_frames)
        if self.save_animation:
            print('\n\nSaving Animation... Please Wait...')

            from datetime import datetime
            time_begin = datetime.now()  # current date and time
            date = time_begin.strftime("%m_%d_%Y")
            name = f'{self.save_dir}\\Outputs\\{date}_Simulation_{data_name}'
            self.anim.save(f'{name}.mp4', writer=self.writer)
            print(f'Animation Saved to: {name}')
        if self.show_animation:
            print(f'\n\nStarting Simulation...')
            plt.show()

class Quat2JointAngles:
    def __init__(self):
        root_dir = 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files'
        self.root_dir = root_dir
        self.processed_dir=self.root_dir+'\\ProcessedData'
        self.raw_dir = self.root_dir + '\\RawData'
        self.output_dir = self.root_dir+'\\Outputs'

        # Joint Angle Processing Variables
        self.plot_processed = True
        self.save_processed = False,
        self.filtered = True
        self.filter_iter = 7
        self.filt_denom = 1

        self.vel_scale = 20
        self.acc_scale = 20

    def quat2KneeHipShank(self, quat_csv_path, init_idxs,name_prefix='',
                          plot_type='line',
                          derivatives = True):
        """
        Input Variables:
            init_idxs: list of length n_lap describing indexs where knee was set to zero displacement
            first_steps: list of length n_lap of either <'flexion' or 'extension'> defining dynamcis of first step
        Where
            n_lap: the number of laps walked by the patient
        """

        from Modules.Utils import Data_Filters
        import matplotlib.pyplot as plt
        from Modules.Utils.typeAtools.formatted_outputs import printh

        # Import Quat data
        data = genfromtxt(quat_csv_path, delimiter=',')
        init_idxs.append(len(data))

        printh(0, 'Begining [Quat] to [Knee,Hip,Shank] transformation...')
        printh(1, f'init_idxs (n={len(init_idxs) - 1} + 1): {init_idxs}')

        # Initialize arrays
        null_vec = [1, 0, 0]
        uninit_data = np.zeros(init_idxs[0]).tolist()
        knee_angles = [] + uninit_data
        hip_angles = [] + uninit_data
        shank_angles = [] + uninit_data

        ##################################
        # ITERATE THROUGH ALL WALKING LAPS
        ##################################
        printh(0, 'Lap Outputs')
        printh(1, f'Lap 0 (Before Walking)')
        printh(2, f'Idx Range = [0,{init_idxs[0]}]')
        printh(2, f'Lap Size = {len(hip_angles)}')

        for idx in range(len(init_idxs[:-1])):

            # Define new reference rotation
            init_quats = data[init_idxs[idx], :]
            #if is_inverted: init_quats[4:] = Quaternion(init_quats[4:]).normalised.inverse.elements
            # ITERATE THROUGH ALL SAMPLES IN THIS WALKING LAP
            for i in range(init_idxs[idx], init_idxs[idx + 1]):

                # Get this quaternion rotation and vectors
                quats = data[i, :]
                #if is_inverted: quats[4:] = Quaternion(data[i, 4:]).normalised.inverse.elements

                thigh_quat, femur_quat = self.quat2KneeHipShank_initialize_quats(quats, init_quats)
                thigh_vec = thigh_quat.rotate(null_vec)
                shank_vec = femur_quat.rotate(null_vec)

                #knee = self.quat2KneeHipShank_AcuteAngBetweenV(thigh_vec, shank_vec)
                #knee = self.quat2KneeHipShank_angBetweenV(thigh_vec, shank_vec,shank_vec)
                #hip = self.quat2KneeHipShank_angBetweenV(thigh_vec, shank_vec,null_vec)
                #shank = self.quat2KneeHipShank_angBetweenV(shank_vec,thigh_vec,null_vec)
                knee,hip,shank = self.quat2KneeHipShank_vec2KHSangle(thigh_vec, shank_vec)

                hip_angles.append(hip)
                shank_angles.append(shank)
                knee_angles.append(knee)

                #printh(3, f'Iteration Size = {len(hip_angles)}\t Hip = {hip}')
                #sleep(0.5)

            printh(1, f'Lap {idx+1}')
            printh(2, f'Idx Range = [{init_idxs[idx]},{init_idxs[idx + 1]}]')
            printh(2, f'Expected Lap Size = {init_idxs[idx + 1]-init_idxs[idx]}')
            printh(2, f'Cumulative Size = {len(hip_angles)}')

        printh(0, 'Final Results')
        printh(1, f'Num Processed:\t n={len(knee_angles)}')
        printh(1, f'Num Samples:\t n={len(data)}')
        printh(1, f'Sample Angle:\t {hip}')

        ##################################
        # FILTER
        ##################################
        if self.filtered:

            knee_angles = Data_Filters.lfilter(knee_angles, self.filter_iter,denom=self.filt_denom)
            hip_angles = Data_Filters.lfilter(hip_angles, self.filter_iter,denom=self.filt_denom)
            shank_angles = Data_Filters.lfilter(shank_angles, self.filter_iter,denom=self.filt_denom)

        ##################################
        # CALC DERIVATIVES
        ##################################
        if derivatives:
            filter_iter = 5
            angles = hip_angles
            hv = Data_Filters.lfilter(np.gradient(angles), filter_iter, denom=self.filt_denom)
            ha = Data_Filters.lfilter(np.gradient(hv), filter_iter, denom=self.filt_denom)

            angles = knee_angles
            kv = Data_Filters.lfilter(np.gradient(angles), filter_iter, denom=self.filt_denom)
            ka = Data_Filters.lfilter(np.gradient(kv), filter_iter, denom=self.filt_denom)

            angles = shank_angles
            sv = Data_Filters.lfilter(np.gradient(angles),filter_iter, denom=self.filt_denom)
            sa = Data_Filters.lfilter(np.gradient(sv), filter_iter, denom=self.filt_denom)

        ##################################
        # PLOT
        ##################################
        if self.plot_processed:
            # Plotting figures
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax_kneeAngles = fig.add_subplot(3, 1, 1)
            ax_hipAngles = fig.add_subplot(3, 1, 2)
            ax_shankAngles = fig.add_subplot(3, 1, 3)

            ax_kneeAngles.set_title(f'Knee Angles | Filtered:{self.filtered}')
            ax_hipAngles.set_title(f'Hip Angles | Filtered:{self.filtered}')
            ax_shankAngles.set_title(f'Shank Angles | Filtered:{self.filtered}')

            ax_kneeAngles.vlines(init_idxs, min(knee_angles), max(knee_angles), colors='r', linestyles=':')
            ax_hipAngles.vlines(init_idxs, min(hip_angles), max(hip_angles), colors='r', linestyles=':')
            ax_shankAngles.vlines(init_idxs, min(shank_angles), max(shank_angles), colors='r', linestyles=':')

            if plot_type=='line':

                ax_hipAngles.plot(hip_angles, linewidth=1, label='t')
                ax_hipAngles.plot(self.vel_scale * hv, linewidth=1, label=f"{self.vel_scale}x dt")
                ax_hipAngles.plot(self.acc_scale * ha, linewidth=1, label=f"{self.acc_scale}x ddt")
                ax_hipAngles.legend(loc = 'upper right')

                ax_shankAngles.plot(shank_angles, linewidth=1, label='t')
                ax_shankAngles.plot(self.vel_scale * sv, linewidth=1, label=f"{self.vel_scale}x dt")
                ax_shankAngles.plot(self.acc_scale * sa, linewidth=1,label=f"{self.acc_scale}x ddt")
                ax_shankAngles.legend(loc = 'upper right')

                ax_kneeAngles.plot(knee_angles, linewidth=1, label="t")
                ax_kneeAngles.plot(self.vel_scale * kv, linewidth=1, label=f"{self.vel_scale}x dt")
                ax_kneeAngles.plot(self.acc_scale * ka, linewidth=1,
                                   label=f"{self.acc_scale}x ddt")
                ax_kneeAngles.legend(loc = 'upper right')

            else:
                ax_kneeAngles.scatter(np.arange(0, len(knee_angles)), knee_angles, s=1)
                ax_hipAngles.scatter(np.arange(0, len(hip_angles)), hip_angles, s=1)
                ax_shankAngles.scatter(np.arange(0, len(shank_angles)), shank_angles, s=1)

            ax_kneeAngles.plot([0 for pt in range(len(knee_angles))], c='k', linewidth=1)
            ax_hipAngles.plot([0 for pt in range(len(hip_angles))], c='k', linewidth=1)
            ax_shankAngles.plot([0 for pt in range(len(shank_angles))], c='k', linewidth=1)



            plt.subplots_adjust(right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5)



        ##################################
        # PACKAGE
        ##################################
        knee_angles = np.reshape(knee_angles, (-1, 1))
        kv = np.reshape(kv, (-1, 1))
        ka = np.reshape(ka, (-1, 1))

        hip_angles = np.reshape(hip_angles, (-1, 1))
        hv = np.reshape(hv, (-1, 1))
        ha = np.reshape(ha, (-1, 1))

        shank_angles = np.reshape(shank_angles, (-1, 1))
        sv = np.reshape(sv, (-1, 1))
        sa = np.reshape(sa, (-1, 1))


        dKH = np.append(kv, hv, axis=1)
        dKHS = np.append(dKH, sv, axis=1)
        ddKH = np.append(ka, ha, axis=1)
        ddKHS = np.append(ddKH, sa, axis=1)

        KH = np.append(knee_angles, hip_angles, axis=1)
        KHS = np.append(KH, shank_angles, axis=1)
        KHS = np.append(KHS,dKHS,axis=1)
        KHS = np.append(KHS, ddKHS, axis=1)

        if self.save_processed:
            from datetime import datetime
            time_begin = datetime.now()  # current date and time
            date = time_begin.strftime("%m_%d_%Y")
            if name_prefix != '': name_prefix = name_prefix + '_'
            name = f'{self.output_dir}\\{date}_{name_prefix}Quat2KneeHipShank'
            savetxt(name + '.csv', KHS, delimiter=',')
            if self.plot_processed: plt.savefig(name + ".png", dpi=600)
            print('Saved Output:\n', name)

        if self.plot_processed:
            print('Plotting output. Exit plot to continue...')
            plt.show()



        return KHS


    def quat2KneeHipShank_initialize_quats(self,quats, init_quats, start_displace=0):

        thigh_quat = quats[0:4]
        femur_quat = quats[4:8]
        thigh_quat = Quaternion(thigh_quat).normalised
        femur_quat = Quaternion(femur_quat).normalised

        thigh_init_quat = init_quats[0:4]
        femur_init_quat = init_quats[4:8]
        thigh_init = self.quat2KneeHipShank_rel_rot(Quaternion(thigh_init_quat))
        femur_init = self.quat2KneeHipShank_rel_rot(Quaternion(femur_init_quat))

        init_displace_quat_thigh = Quaternion(axis=thigh_quat.axis, degrees=start_displace / 2.)
        init_displace_quat_femur = Quaternion(axis=femur_quat.axis, degrees=start_displace / 2.)

        thigh_quat = thigh_init * thigh_quat
        thigh_quat = init_displace_quat_thigh * thigh_quat

        femur_quat = femur_init * femur_quat
        thigh_quat = -init_displace_quat_femur * thigh_quat

        return thigh_quat, femur_quat

    def quat2KneeHipShank_vec2KHSangle(self,V_thigh, V_shank):
        """
        Calculate the angle between vectors about the axis normal to the plane formed by leg
        (except for knee which is minimum angle)
        """

        uv_thigh = V_thigh / np.linalg.norm(np.array(V_thigh))
        uv_shank = V_shank / np.linalg.norm(np.array(V_shank))
        uv_null = np.array([1, 0, 0])

        # Calculate normal vector formed by leg (ei assumed saggital plane)
        hip = np.array([0, 0, 0])
        knee = hip + uv_thigh
        ankle = uv_thigh + uv_shank
        cross =np.cross((knee - hip), (ankle - hip))
        uv_norm = cross / np.linalg.norm(np.array(cross))

        # calculate hip angle
        vecs = [uv_thigh, uv_null]
        dot_product = np.dot(vecs[0], vecs[1])
        det = np.linalg.det(np.append(np.append(vecs[0], vecs[1],axis=0),uv_norm,axis=0).reshape(3,3))
        hip_angle = degrees(np.arctan2(det, dot_product))

        # calculate shank angle
        vecs = [uv_shank, uv_null]
        dot_product = np.dot(vecs[0], vecs[1])
        det = np.linalg.det(np.append(np.append(vecs[0], vecs[1],axis=0),uv_norm,axis=0).reshape(3,3))
        shank_angle = degrees(np.arctan2(det, dot_product))

        # calculate knee angle
        vecs = [uv_thigh, uv_shank]
        dot_product = np.dot(vecs[0], vecs[1])
        det = np.linalg.det(np.append(np.append(vecs[0], vecs[1], axis=0), uv_norm, axis=0).reshape(3, 3))
        knee_angle = degrees(np.arctan2(det, dot_product))

        return knee_angle,hip_angle,shank_angle

    def quat2KneeHipShank_angBetweenV(self,V1, V2, Vref):
        # error handling
        uv1 = V1 / np.linalg.norm(np.array(V1))
        uv2 = V2 / np.linalg.norm(np.array(V2))
        uvref = Vref / np.linalg.norm(np.array(Vref))

        #v1= np.array(v1)
        #v2 = np.array(v2)
        #vref = np.array(vref)

        # Calculate normal vector formed by leg (ei assumed saggital plane)
        hip = np.array([0, 0, 0])
        knee = hip + uv1
        ankle = uv1 + uv2
        cross =np.cross((knee - hip), (ankle - hip))
        uv_norm = cross / np.linalg.norm(np.array(cross))

        # calculate angel
        uv1 = uv1/ np.linalg.norm(uv1)
        uv2 = uvref/ np.linalg.norm(uvref)
        dot_product = np.dot(uv1, uv2)
        det = np.linalg.det(np.append(np.append(uv1,uv2,axis=0),uv_norm,axis=0).reshape(3,3))
        angle = degrees(np.arctan2(det, dot_product))

        return angle

    def quat2KneeHipShank_rel_rot(self,quat, target_quat=Quaternion([1., 0., 0., 0.])):
        quat_init = target_quat * quat.inverse
        quat_init = quat_init.normalised
        return quat_init

    def quat2KneeHipShank_derivs(self,X):
        #X=X.tolist()
        n=len(X)

        nullval = 0.0
        vel = []
        acc = [0,0]
        print(f'Data Shape: {np.shape(X)}')

        for i in range(n):
            if i == 0: v = nullval
            elif i ==1: v = X[1]-X[0]
            elif i == n-1: v = X[i]-X[i-1]

            else: v = (X[i+1]-X[i-1])/2.0
            vel.append(v)
        print(f'Vel Shape: {np.shape(vel)}')
        for i in range(n):
            if i == 0 or i== 1: a = nullval
            elif i ==2: a = vel[2]-vel[1]
            elif i == n-1: a = vel[i]-vel[i-1]
            else: a = (vel[i+1]-vel[i-1])/2.0
            acc.append(a)

        return vel,acc
if __name__ == "__main__":
    path = 0  # named int, do not change
    init_idxs = 1  # named int, do not change
    filter_iter = 2  # named int, do not change

    data_param = [
        #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal1.csv', [200, 1431, 2562, 3601], 10
        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal2.csv',      [268,2966,5349,7535],10
        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal3_180.csv',  [429,3280,6838,9434],10
        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_MocapTrial.csv',  [3176],10

        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S1.csv', [624],10 #[624, 5390, 9453, 16523, 19910]
        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S2.csv', [0],10
        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S3.csv', [11150, 18580, 40731],10

        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S1.csv', [0],10
        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S2.csv', [2210,10364,13051,20501],10
        # 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S3.csv', [560,2451],10
    ]
    data_name = data_param[path].split("\\")[-1].split(".")[0]


    Quat2JointAngles = Quat2JointAngles()
    KHS= Quat2JointAngles.quat2KneeHipShank(quat_csv_path=data_param[path],
                                                init_idxs=data_param[init_idxs],
                                                # filter_iter=data_param[filter_iter],
                                                name_prefix=data_name,
                                                # plot_output=plot_import,
                                                # save_output=save_import,
                                                # filtered=filtered_import
                                                )

    Sim = GaitSimulation(KHS,ffmpeg_path = 'C:\\ffmpeg\\ffmpeg-20200831-4a11a6f-win64-static\\bin\\ffmpeg.exe')
    Sim.index_speed = 2
    Sim.index_start = data_param[init_idxs][0]
    Sim.window_range = 800  # How many indexes in the future in view
    Sim.data_name = data_name
    Sim.run()
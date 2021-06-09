from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import serial
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy import ndimage
from time import sleep

n_sensors = 28
noise_Thresh = 0.2  # Percent of maximimum force reading considered to be zero
#Initiate Serial
ser = serial.Serial('COM3', 115200)

#Initialize Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
CMAP=True
BOTH_SENSORS = False
FILT_NOISE = True
#Initialize sensor value array
data = [10, 0, 0, 0, 0, 0,   # first sensor
        0, 0, 0, 0, 0, 0,   # first sensor
        0, 0, 0, 0, 0,10,   # first sensor
        0, 0, 0, 0, 0,      # second sensor
        0, 0, 0, 0, 0,      # second sensor
        ]
old_data = data
#
# data_s1 = np.array([[2,0,0,0,0,1],
#                      [0,0,0,0,0,0],
#                      [1,0,0,0,0,1]])
# data_s2 = np.array([[2,0,0,0,0,1],
#                      [0,0,0,0,0,0],
#                      [1,0,0,0,0,1]])

def readSer():
    global data
    global n_sensors


    if ser.in_waiting > 0:
        #line = ser.readline()   #read Arduino serial data
        #trash = ser.flush()     #clear lagging data in serial
        sleep(0.1)
        try:
            #Convert to list of integers
            bytes = ser.read_all()

            bytes_str = bytes.decode()

            bytes_str_lst = bytes_str.split("\r")
            #print(np.shape(bytes_str_lst))
            b_str = bytes_str_lst[-2]
            #print(b_str)
            list_str = b_str.split(",")[0:-1]
            #print(np.shape(list_str))
            list_float = [float(item) for item in list_str]
            #print(list_float)
            return list_float
            # line = line.decode()
            # line = line.split(",")
            # data = []
            # if len(line) == n_sensors + 1:
            #     for i in range(n_sensors):
            #         val = int(line[i])
            #         data.append(val)
            # else:
            #     print("unread")
            # return data
        except:
            print("Serial Error")

def centroid(data):
    #data_s1 = np.array([[]])
    num_s1 = 18
    height_s1 =3
    width_s1 = 6

    data_s2 = np.array([])
    num_s2 = 10
    hight_s2 =2
    width_s2 = 5

    i = 0
    data_s1 = np.array(data[:-num_s2]).reshape(-1, 6)
    data_s2 = np.array(data[-num_s2:]).reshape(-1, 5)
    #print(f'Shape of Datas: S1 {np.shape(data_s1)}, S2 {np.shape(data_s2)}')

    cm_s1 = ndimage.measurements.center_of_mass(data_s1)
    mag_s1 = np.sum(data_s1)

    cm_s2 = ndimage.measurements.center_of_mass(data_s2)
    mag_s2 = np.sum(data_s2)

    #data_s2 = np.array(data[:num_s2]).reshape(-1, 5)

    return cm_s1,mag_s1,cm_s2,mag_s2




def updateDraw(i):
    global data
    global old_data
    global noise_Thresh
    gridRes = 0.7
    # Plot figure

    x_loc = [1, 2, 3, 4, 5, 6,  # first sensor
             1, 2, 3, 4, 5, 6,  # first sensor
             1, 2, 3, 4, 5, 6
             ]
    y_loc = [1, 1, 1, 1, 1, 1,  # first sensor
             2, 2, 2, 2, 2, 2,  # first sensor
             3, 3, 3, 3, 3, 3
             ]
    #Both Sensors
    if BOTH_SENSORS:
        x_loc = [1, 2, 3, 4, 5, 6,  # first sensor
                 1, 2, 3, 4, 5, 6,  # first sensor
                 1, 2, 3, 4, 5, 6,
                 1, 2, 3, 4, 5,  # second sensor
                 1, 2, 3, 4, 5  # second sensor
                 ]
        y_loc = [1, 1, 1, 1, 1, 1,  # first sensor
                 2, 2, 2, 2, 2, 2,  # first sensor
                 3, 3, 3, 3, 3, 3,
                 5, 5, 5, 5, 5,  # second sensor
                 6, 6, 6, 6, 6  # second sensor
                 ]

    try:
        data = readSer()
        #print(f'Len(ser)={len(data)}')

        data_display = data[:-10]
        if FILT_NOISE:
            for i in range(len(data)):
                if data[i] < max(data) * noise_Thresh or data[i] <= 2:
                    data[i] = 0.0

        #Append Centroid

        cm_s1, mag_s1,cm_s2, mag_s2 = centroid(data)
        #cm_s1[1] = cm_s1[1]+1
        #cm_s1[0] = cm_s1[0]+ 1
        x_loc.append(cm_s1[1]+1)
        y_loc.append(cm_s1[0]+1)
        data_display.append(mag_s1)

        z_loc = np.zeros(len(x_loc))
        dx = np.ones(len(x_loc))*gridRes
        dy = np.ones(len(x_loc))*gridRes
        #Plot Data
        ax.clear()
        ax.set_xlim([1, 7])
        ax.set_ylim([1, 7])
        ax.set_zlim([0, 150])
        color_values = []


        # Cmap
        if CMAP:
            length = len(x_loc)
            cmap = cm.get_cmap('jet')  # Get desired colormap
            max_height = np.max(data_display[0:length-1])  # get range of colorbars
            min_height = np.min(data_display[0:length-1])

            # scale each z to [0,1], and get their rgb values
            rgba = [cmap((k - min_height) / max_height) for k in data_display[0:length-1]]
            #color_values = cm.jet(norm(fracs.tolist()))

        try:
            length = len(x_loc)
            ax.bar3d(x_loc[0:length-1], y_loc[0:length-1], z_loc[0:length-1],
                     dx[0:length-1], dy[0:length-1], data_display[0:length-1],
                     color = rgba)
            ax.bar3d(x_loc[- 1], y_loc[- 1], z_loc[- 1], dx[- 1],
                     dy[- 1], data_display[- 1], color = 'red')

        except: #Error Handling for bad data
            data = old_data

            # # Polt
            # print(mag_s1,"at",cm_s1)
            # print("X:",len(x_loc))
            # print("Y:",len(y_loc))
            # print("Z:",len(data_display))
            # print(x_loc)
            # print(y_loc)
            # print(data_display)
            # print(z_loc)
            # print(dx)
            # print(dy)
            #length = len(x_loc)
            #ax.bar3d(x_loc[0:length - 1], y_loc[0:length - 1], z_loc[0:length - 1],
            #         dx[0:length - 1], dy[0:length - 1], data_display[0:length - 1],
            #         color=rgba)
            #ax.bar3d(x_loc[- 1], y_loc[- 1], z_loc[- 1], dx[- 1],
            #         dy[- 1], data_display[- 1], color='red')
            ax.bar3d(x_loc, y_loc, z_loc, dx, dy, data, color=color_values)

        print(mag_s1, "at", cm_s1)
    except:
        print("error")
# x_loc = []
# y_loc = []
# dz = []
# for r in range(len(data_s1)):
#     for c in range(len(data_s1[0])):
#         x_loc.append(c)
#         y_loc.append(r)
#         dz.append(data_s1[r][c])
#
# cm = ndimage.measurements.center_of_mass(data_s1)
# mag = np.sum(data_s1)



#
# gridRes = 0.5
# dx = np.ones(len(x_loc))*gridRes
# dy = np.ones(len(x_loc))*gridRes
# z_loc =  np.zeros(len(dz))
#
# #print(mag,"at",cm)
# #print("X:",x_loc)
# #print("Y:",y_loc)
# #print("Z:",dz)
#
# ax.bar3d(x_loc, y_loc, z_loc, dx, dy, dz)
# #for point in range(len)
a = anim.FuncAnimation(fig, updateDraw, interval=1)
plt.show()

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import serial
import matplotlib.colors as colors
import matplotlib.cm as cm
from time import sleep
import matplotlib.cm as cm
from scipy import ndimage

n_sensors = 28
noise_Thresh = 0.2  # Percent of maximimum force reading considered to be zero
#Initiate Serial
ser = serial.Serial('COM3', 115200)

#Initialize Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Initialize sensor value array
data = [0, 0, 0, 0, 0, 0,   # first sensor
        0, 0, 0, 0, 0, 0,   # first sensor
        0, 0, 0, 0, 0, 0,   # first sensor
        0, 0, 0, 0, 0,      # second sensor
        0, 0, 0, 0, 0,      # second sensor
        ]

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
    data_s1 = np.array(data[:-num_s1]).reshape(-1, 6)
    data_s2 = np.array(data[num_s1:num_s2]).reshape(-1, 5)
    #print(f'Shape of Datas: S1 {np.shape(data_s1)}, S2 {np.shape(data_s2)}')

    cm_s1 = ndimage.measurements.center_of_mass(data_s1)
    mag_s1 = np.sum(data_s1)

    cm_s2 = ndimage.measurements.center_of_mass(data_s2)
    mag_s2 = np.sum(data_s2)

    #data_s2 = np.array(data[:num_s2]).reshape(-1, 5)

    return cm_s1,mag_s1,cm_s2,mag_s2



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

def updateDraw(i):
    global data
    global noise_Thresh

    x_loc = [1, 2, 3, 4, 5, 6,  # first sensor
             1, 2, 3, 4, 5, 6,  # first sensor
             1, 2, 3, 4, 5, 6,  # first sensor
             1, 2, 3, 4, 5,     # second sensor
             1, 2, 3, 4, 5      # second sensor
             ]
    y_loc = [1, 1, 1, 1, 1, 1,  # first sensor
             2, 2, 2, 2, 2, 2,  # first sensor
             3, 3, 3, 3, 3, 3,  # first sensor
             5, 5, 5, 5, 5,     # second sensor
             6, 6, 6, 6, 6      # second sensor
             ]


    oldData = data
    data = readSer()

    #Plot Data
    ax.clear()
    ax.set_xlim([1, 7])
    ax.set_ylim([1, 7])
    ax.set_zlim([0, 150])
    try:
        for i in range(len(data)):
            if data[i] < max(data) * noise_Thresh or data[i]<=2:
                data[i] = 0.1
        # Cmap
        CENTROID = False
        if CENTROID:
            cm_s1, mag_s1, cm_s2, mag_s2 = centroid(data)
            print(cm_s1, mag_s1, cm_s2, mag_s2)
            x_loc.append(cm_s1[0]).append(cm_s2[0])
            y_loc.append(cm_s1[1]).append(cm_s2[1])
            #z_loc.append(0, 0)
            #dx.append(w, w)
            #dy.append(h, h)
            data.append(mag_s1, mag_s2)
            #color_values.append(color_values[-1], color_values[-1])

        offset = data + np.abs(min(data))
        fracs = offset.astype(float) / max(offset)
        norm = colors.Normalize(min(fracs), max(fracs))
        color_values = cm.jet(norm(fracs.tolist()))
        z_loc = np.zeros(len(x_loc))
        dx = np.ones(len(x_loc))
        dy = np.ones(len(x_loc))
        ax.bar3d(x_loc, y_loc, z_loc, dx, dy, data, color=color_values)

        #cm_s1, mag_s1, cm_s2, mag_s2 = centroid(data)
        #ax.bar3d([cm_s1[0],cm_s2[0]], [cm_s1[1],cm_s2[1]],
        #         np.zeros(2), 0.5*np.ones(2), 0.5*np.ones(2), 3*[mag_s1,mag_s2], color=color_values)

    except: #Error Handling for bad data
        data = oldData
        for i in range(len(data)):
            if data[i] < max(data) * noise_Thresh or data[i]<=2:
                data[i] = 0.1
        # Cmap
        CENTROID = False
        if CENTROID:
            cm_s1, mag_s1, cm_s2, mag_s2 = centroid(data)
            print(cm_s1, mag_s1, cm_s2, mag_s2)
            x_loc = x_loc + [cm_s1[0], cm_s2[0]]
            y_loc = y_loc + [cm_s1[1], cm_s2[1]]
            # z_loc.append(0, 0)
            # dx.append(w, w)
            # dy.append(h, h)
            data = data  +[mag_s1, mag_s2]
            # color_values.append(color_values[-1], color_values[-1])

        offset = data + np.abs(min(data))
        fracs = offset.astype(float) / max(offset)
        norm = colors.Normalize(min(fracs), max(fracs))
        color_values = cm.jet(norm(fracs.tolist()))
        z_loc = np.zeros(len(x_loc))
        dx = np.ones(len(x_loc))
        dy = np.ones(len(x_loc))
        # Polt
        ax.bar3d(x_loc, y_loc, z_loc, dx, dy, data, color=color_values)
    #print(color_values)

a = anim.FuncAnimation(fig, updateDraw, interval=1)
plt.show()

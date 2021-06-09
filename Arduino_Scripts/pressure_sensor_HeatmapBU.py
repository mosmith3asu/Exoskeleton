from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import serial
import matplotlib.colors as colors
import matplotlib.cm as cm
from Modules.Utils.Import_Utilis import serial_legsensor
n_sensors = 28
noise_Thresh = 0.2  # Percent of maximimum force reading considered to be zero
#Initiate Serial
#ser = serial.Serial('COM3', 115200)
leg_ser = serial_legsensor('COM3')
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


def readSer():
    global data
    global n_sensors


    if ser.in_waiting > 0:
        line = ser.readline()   #read Arduino serial data
        trash = ser.flush()     #clear lagging data in serial
        try:
            #Convert to list of integers
            line = line.decode()
            line = line.split(",")
            data = []
            if len(line) == n_sensors + 1:
                for i in range(n_sensors):
                    val = int(line[i])
                    data.append(val)
            else:
                print("unread")
            return data
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
    z_loc = np.zeros(len(x_loc))

    dx = np.ones(len(x_loc))
    dy = np.ones(len(x_loc))

    oldData = data
    #data = readSer()
    data = leg_ser.read_line()
    print()

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
        offset = data + np.abs(min(data))
        fracs = offset.astype(float) / max(offset)
        norm = colors.Normalize(min(fracs), max(fracs))
        color_values = cm.jet(norm(fracs.tolist()))
        ax.bar3d(x_loc, y_loc, z_loc, dx, dy, data, color=color_values)

    except: #Error Handling for bad data
        data = oldData
        for i in range(len(data)):
            if data[i] < max(data) * noise_Thresh or data[i]<=2:
                data[i] = 0.1
        # Cmap
        offset = data + np.abs(min(data))
        fracs = offset.astype(float) / max(offset)
        norm = colors.Normalize(min(fracs), max(fracs))
        color_values = cm.jet(norm(fracs.tolist()))
        # Polt
        ax.bar3d(x_loc, y_loc, z_loc, dx, dy, data, color=color_values)
    print(color_values)

a = anim.FuncAnimation(fig, updateDraw, interval=1)
plt.show()

# Analyzing 2 Dimiensions = Knee Angle and Gait Phase

from numpy import genfromtxt,savetxt
from math import *
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from scipy.interpolate import interp1d

SAVE_MODEL=True

def expected_knee_angle(gait_phase):
    expected_angle = 1e-08*pow(gait_phase,6) - 4e-06*pow(gait_phase,5) + 0.0003*pow(gait_phase,4) - 0.0112*pow(gait_phase,3) + 0.1127*pow(gait_phase,2) + 0.5405*gait_phase + 9.9107
    return expected_angle

data_paths = [
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L1.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L2.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L3.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L1.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L2.csv",
    "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L3.csv"
]
compare2healthy = False
# Import Data

paired_data = genfromtxt(data_paths[0], delimiter=',')
data_S3L1 = paired_data[1:,0:3]
regression_data = data_S3L1
print(max(data_S3L1[:,0]))
if compare2healthy:
    for i in range(len(regression_data)):
        gaitphse = regression_data[i,0]
        print("Old angle", regression_data[i, 1])
        regression_data[i,1] = expected_knee_angle(gaitphse)-regression_data[i,1]
        print("New angle",regression_data[i,1])

#Ref https://stackoverflow.com/questions/30696741/how-to-implement-kernel-density-estimation-in-multivariate-3d
data=data_S3L1

data = data.T #The KDE takes N vectors of length K for K data points
              #rather than K vectors of length N
#data = true_data
print("Patient_Data Shape ",np.shape(data))
kde = stats.gaussian_kde(data)

# You now have your kde!!  Interpreting it / visualising it can be difficult with 3D data
# You might like to try 2D data first - then you can plot the resulting estimated pdf
# as the height in the third dimension, making visualisation easier.

# Here is the basic way to evaluate the estimated pdf on a regular n-dimensional mesh
# Create a regular N-dimensional grid with (arbitrary) 20 points in each dimension
minima = data.T.min(axis=0)
maxima = data.T.max(axis=0)
space = [np.linspace(mini,maxi,20) for mini, maxi in zip(minima,maxima)]
grid = np.meshgrid(*space)

#Turn the grid into N-dimensional coordinates for each point
#Note - coords will get very large as N increases...
coords = np.vstack(map(np.ravel, grid))

#Evaluate the KD estimated pdf at each coordinate
density = kde(coords)

#Do what you like with the density values here..
#plot them, output them, use them elsewhere...
fig = plt.figure()
coords = coords.T

maxD = np.max(density)
minD = np.min(density)

#Map sizes to density value
size_interp = interp1d([minD,maxD],[3,8])
sizes=size_interp(density)

#Map color & alpha to density value
norm = colors.Normalize(vmin=minD, vmax=maxD, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
rgba = []

for d in range(len(density)):
    val = mapper.to_rgba(density[d])
    r,g,b,a = val
    a = density[d]/maxD
    val = (r,g,b,a)
    rgba.append(val)


ax = fig.gca(projection='3d')
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],s=sizes,c=rgba)#c=density,
#ax.plot(true_data[:, 0], true_data[:, 1], true_data[:, 2])
ax.scatter(data[:, 0], data[:, 1],data[:, 2],s=8,c="black",marker="v")
surf = ax.plot_surface(coords[:, 0], coords[:, 1], coords[:, 2], rstride=1, cstride=1, alpha=0.5)# cmap=cm.jet,
#surf = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], linewidth=0, antialiased=False)
plt.xlabel('Gait Phase %')
plt.ylabel('Knee Angle Error')
ax.set_zlabel('Torque Applied')
plt.title("GMM For Patient3S1 Lap1")

ax.axis('tight')


plt.show()
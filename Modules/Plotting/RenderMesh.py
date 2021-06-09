from math import *

import numpy as np
from scipy.linalg import norm
from scipy.spatial.transform import Rotation
from stl import mesh
from Modules.Utils import Import_Utilis

mesh_leg0 = []
# Load the STL files and add the vectors to the plot
this_file_dir = Import_Utilis.get_this_dir(__file__)
thigh_mesh_dir = this_file_dir + '\STL_Thigh.stl'
femur_mesh_dir = this_file_dir + '\STL_LowerLeg.stl'

# Add meshes to list
mesh_leg0.append(mesh.Mesh.from_file(thigh_mesh_dir))
mesh_leg0.append(mesh.Mesh.from_file(femur_mesh_dir))
# Serial
# import serial
# ser = serial.Serial("COM9", 115200)
# sleep(1)

# GLOBAL VARS
x_axis = [1.0, 0.0, 0.0]
y_axis = [0.0, 1.0, 0]
z_axis = [0.0, 0, 1.0]


def rotM_vec(rotM, vec):
    r = Rotation.from_matrix(rotM)
    rot_vec = r.apply(vec)
    return rot_vec


def normalize_rotM(quat):
    r = Rotation.from_quat(quat, normalized=True)
    return r.as_quat()


def normalize_quat(quat):
    r = Rotation.from_quat(quat, normalized=True)
    return r.as_quat()


def inv_quat(quat):
    r1 = Rotation.from_quat(quat)
    r2 = r1.inv()
    return r2.as_quat()


def inv_rotM(rotM):
    r1 = Rotation.from_matrix(rotM)
    r2 = r1.inv()
    return r2.as_matrix()


def angBetweenRotM(rotM1, rotM2):
    r1 = Rotation.from_matrix(rotM1)
    r2 = Rotation.from_matrix(rotM2)

    v1 = r1.as_rotvec()
    v2 = r2.as_rotvec()

    return angBetweenV(v1, v2)


def angBetweenQuat(quat1, quat2):
    r1 = Rotation.from_quat(quat1)
    r2 = Rotation.from_quat(quat2)

    v1 = r1.as_rotvec()
    v2 = r2.as_rotvec()

    return angBetweenV(v1, v2)


def angBetweenV(v1, v2):
    # check if there is an array of zeros at hip
    is_all_zero1 = np.all((v1 == 0))
    if is_all_zero1:
        v1 = np.array([1., 0, 0])

    # calculate angel
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = degrees(np.arccos(dot_product))
    return angle

def quat2ypr(rotation_Quat,return_in_degrees=True):
    rot = Rotation.from_quat(rotation_Quat)
    return rot.as_euler('zyx',degrees=return_in_degrees)

def ypr2rotM(rotation_YPR, in_degrees=True, debug=False):
    if in_degrees:
        for i in range(len(rotation_YPR)):
            rotation_YPR[i] = radians(rotation_YPR[i])

    rotation = Rotation.from_euler('zyx', rotation_YPR)
    rot_M = Rotation.as_matrix(rotation)

    if debug:
        print(rot_M)

    return rot_M


def ypr2quat(rotation_YPR, in_degrees=True, debug=False):
    if in_degrees:
        for i in range(len(rotation_YPR)):
            rotation_YPR[i] = radians(rotation_YPR[i])

    rotation = Rotation.from_euler('zyx', rotation_YPR)
    quat = Rotation.as_quat(rotation)

    if debug:
        print(quat)

    return quat


def quat2rotM(rotion_QUAT, return_in_degrees=True, debug=False):
    rotation = Rotation.from_quat(rotion_QUAT)
    rot_M = Rotation.as_matrix(rotation)

    if debug:
        print(rot_M)
    # if return_in_degrees:
    #    rot_M=[degrees(rot_M[0]),degrees(rot_M[1]),degrees(rot_M[2])]
    return rot_M


def quat2rotV(rotion_QUAT, return_in_degrees=True, debug=False):
    rotation = Rotation.from_quat(rotion_QUAT)
    rot_M = Rotation.as_rotvec(rotation)
    if debug:
        ("Rotation Matrix: ", degrees(rot_M[0]), degrees(rot_M[1]), degrees(rot_M[2]))
    if return_in_degrees:
        rot_M = [degrees(rot_M[0]), degrees(rot_M[1]), degrees(rot_M[2])]
    return rot_M


def rotatedPoint(vec, rotation_Matrix, in_DEG=True):
    vec = np.array(vec)
    vec_mag = norm(vec)
    vec = vec / vec_mag
    # print(vec)

    # X-Axis
    rotation_DEG = rotation_Matrix[0]
    rotation_axis = -1.0 * np.array(x_axis)

    if in_DEG:
        rotation_RAD = np.radians(rotation_DEG)
    else:
        rotation_RAD = rotation_DEG
    rotation_axis = np.array(rotation_axis)
    rotation_vector = rotation_RAD * rotation_axis
    rotation = Rotation.from_rotvec(rotation_vector)
    vec = rotation.apply(vec)
    # print(vec)

    # Y-Axis
    rotation_DEG = rotation_Matrix[1]
    rotation_axis = -1.0 * np.array(y_axis)

    if in_DEG:
        rotation_RAD = np.radians(rotation_DEG)
    else:
        rotation_RAD = rotation_DEG
    rotation_axis = np.array(rotation_axis)
    rotation_vector = rotation_RAD * rotation_axis
    rotation = Rotation.from_rotvec(rotation_vector)
    vec = rotation.apply(vec)
    # print(vec)

    # z-Axis
    rotation_DEG = rotation_Matrix[2]
    rotation_axis = -1.0 * np.array(z_axis)

    if in_DEG:
        rotation_RAD = np.radians(rotation_DEG)
    else:
        rotation_RAD = rotation_DEG
    rotation_axis = np.array(rotation_axis)
    rotation_vector = rotation_RAD * rotation_axis
    rotation = Rotation.from_rotvec(rotation_vector)
    vec = rotation.apply(vec)
    # print(vec)

    vec = vec * vec_mag
    # print(vec)

    return vec


def rotM2ypr(rotM):
    r = Rotation.from_matrix(rotM)
    return r.as_euler('zyx', degrees=True)


def bounding_box(obj):
    minx = obj.x.min()
    maxx = obj.x.max()
    miny = obj.y.min()
    maxy = obj.y.max()
    minz = obj.z.min()
    maxz = obj.z.max()
    return minx, maxx, miny, maxy, minz, maxz


def generate_sensor_mesh(p0, p1, resolution=6, d_Theta=120, radius=0.12):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # origin = np.array([0, 0, 0])
    # axis and radius
    # p0 = np.array([1, 3, 2])
    # p1 = np.array([8, 5, 9])
    R = radius
    res = resolution
    # vector in direction of axis
    v = np.array(p1) - np.array(p0)
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, resolution)
    # theta = np.linspace(0,  2*np.pi, resolution)
    theta = np.linspace(0, radians(d_Theta), resolution)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    return X, Y, Z


def generate_legmesh_simple(hip_Ang, knee_Ang):
    # hip_Ang = -30
    # knee_Ang = 30
    knee_Ang = hip_Ang - knee_Ang
    # Leg pointers
    mesh_leg = []
    thigh = 0
    lowerLeg = 1

    # Mesh Alignment Constants
    hip = [0, 0, 0]
    knee = [0, 0, 0]
    ankle = [0, 0, 0]
    x_Offset = 0.02
    y_Offset = 0.005

    # Load the STL files and add the vectors to the plot
    try:
        mesh_leg.append(mesh.Mesh.from_file('ThighSTL.stl'))
        mesh_leg.append(mesh.Mesh.from_file('LowerlegSTL.stl'))
    except:
        mesh_leg.append(mesh.Mesh.from_file('Models/ThighSTL.stl'))
        mesh_leg.append(mesh.Mesh.from_file('Models/LowerlegSTL.stl'))

    # Center Hip
    minx, maxx, miny, maxy, minz, maxz = bounding_box(mesh_leg[thigh])
    L_thigh = float(maxz - minz)
    pivot = [(minx + maxx) / 2.0, (miny + maxy) / 2.0, maxz]
    offset = [hip[0] - pivot[0], hip[1] - pivot[1], hip[2] - pivot[2]]
    mesh_leg[thigh].x += offset[0]
    mesh_leg[thigh].y += offset[1]
    mesh_leg[thigh].z += offset[2]

    # Get Lower Leg Length
    minx, maxx, miny, maxy, minz, maxz = bounding_box(mesh_leg[lowerLeg])
    L_lowerLeg = float(maxz - minz)

    # Rotate Parts
    mesh_leg[thigh].rotate([0.5, 0.0, 0.0], radians(hip_Ang))
    mesh_leg[lowerLeg].rotate([0.5, 0.0, 0.0], radians(knee_Ang))

    # Align Rest of leg (initial offsets)
    mesh_leg[lowerLeg].x -= offset[0] + x_Offset
    mesh_leg[lowerLeg].y -= offset[1] + y_Offset
    mesh_leg[lowerLeg].z -= offset[2] + L_thigh

    # Align Rest of leg (rotational offsets)
    follow_y = L_thigh * sin(radians(hip_Ang))
    follow_z = L_thigh * cos(radians(hip_Ang))
    mesh_leg[lowerLeg].y += -follow_y
    mesh_leg[lowerLeg].z += (L_thigh - follow_z)

    # Generate Leg Vectors
    knee = [0, -follow_y, -follow_z]

    follow_y = L_lowerLeg * sin(radians(knee_Ang))
    follow_z = L_lowerLeg * cos(radians(knee_Ang))

    ankle = [0, -follow_y + knee[1], -follow_z + knee[2]]

    return mesh_leg, hip, knee, ankle


def generate_legmesh_Quat(thigh_quat, femur_quat, debug=False):
    #global mesh_leg0
    # Leg pointers
    mesh_leg = mesh_leg0


    thigh = 0
    lowerLeg = 1

    # Mesh Alignment Constants
    hip = [0, 0, 0]



    # Rotate Leg meshes to initial state
    # ypr_thigh = rotM2ypr(thigh_RotM)
    # ypr_femur = rotM2ypr(femer_RotM)
    # mesh_leg[thigh].rotate(z_axis, radians(ypr_thigh[0] + 90))
    # mesh_leg[lowerLeg].rotate(z_axis, radians(ypr_femur[0] + 90))
    # mesh_leg[thigh].rotate(z_axis, radians(-90))
    # mesh_leg[lowerLeg].rotate(z_axis, radians(-90))
    # mesh_leg[thigh].rotate(x_axis, radians(180))
    # mesh_leg[lowerLeg].rotate(x_axis, radians(180))

    # mesh_leg[thigh].rotate(y_axis, radians(-90))
    # mesh_leg[lowerLeg].rotate(y_axis, radians(-90))

    # Center Hip
    minx, maxx, miny, maxy, minz, maxz = bounding_box(mesh_leg[thigh])
    L_thigh = float(maxz - minz)
    pivot = [(minx + maxx) / 2.0, (miny + maxy) / 2.0, maxz]
    offset = [hip[0] - pivot[0], hip[1] - pivot[1], hip[2] - pivot[2]]
    # mesh_leg[thigh].x += offset[0]
    # mesh_leg[thigh].y += offset[1]
    # mesh_leg[thigh].z += offset[2]

    # Find Length of lower leg
    minx, maxx, miny, maxy, minz, maxz = bounding_box(mesh_leg[thigh])
    L_femer = float(maxz - minz)

    # Rotate Parts
    mesh_leg[thigh].rotate(thigh_quat.axis,thigh_quat.radians)

    thigh_Vec = np.array([0, 0, -L_thigh])
    knee = thigh_quat.rotate(thigh_Vec)#np.dot(thigh_Vec, thigh_RotM)

    femur_Vec = np.array([0., 0., -L_femer])
    ankle = femur_quat.rotate(femur_Vec)#np.dot(femer_Vec, femur_RotM)

    # Move the femur to the knee location
    mesh_leg[lowerLeg].x += knee[0]  # + offset[0]
    mesh_leg[lowerLeg].y += knee[1]  # + offset[1]
    mesh_leg[lowerLeg].z += knee[2]  # + offset[2]

    # Rotate around point knee
    mesh_leg[lowerLeg].rotate(femur_quat.axis,femur_quat.radians)

    knee_Ang = angBetweenV(knee, ankle)

    hip_plane_v = np.array([knee[0], knee[1], 0])
    hip_Ang = angBetweenV(hip_plane_v, knee)

    mesh_leg[thigh].rotate(x_axis, radians(180))
    mesh_leg[lowerLeg].rotate(x_axis, radians(180))

    return mesh_leg, hip, knee, ankle, knee_Ang, hip_Ang


def generate_legmesh_Matrix(thigh_RotM, femer_RotM, debug=False):
    # Leg pointers

    thigh = 0
    lowerLeg = 1

    # Mesh Alignment Constants
    hip = [0, 0, 0]

    mesh_leg = []
    # Load the STL files and add the vectors to the plot
    this_file_dir = Import_Utilis.get_this_dir(__file__)
    thigh_mesh_dir = this_file_dir + '\STL_Thigh.stl'
    femur_mesh_dir = this_file_dir + '\STL_LowerLeg.stl'

    # Add meshes to list
    mesh_leg.append(mesh.Mesh.from_file(thigh_mesh_dir))
    mesh_leg.append(mesh.Mesh.from_file(femur_mesh_dir))

    # Rotate Leg meshes to initial state
    #ypr_thigh = rotM2ypr(thigh_RotM)
    #ypr_femur = rotM2ypr(femer_RotM)
    #mesh_leg[thigh].rotate(z_axis, radians(ypr_thigh[0] + 90))
    #mesh_leg[lowerLeg].rotate(z_axis, radians(ypr_femur[0] + 90))
    #mesh_leg[thigh].rotate(z_axis, radians(-90))
    #mesh_leg[lowerLeg].rotate(z_axis, radians(-90))
   # mesh_leg[thigh].rotate(x_axis, radians(180))
   # mesh_leg[lowerLeg].rotate(x_axis, radians(180))

    # mesh_leg[thigh].rotate(y_axis, radians(-90))
    # mesh_leg[lowerLeg].rotate(y_axis, radians(-90))

    # Center Hip
    minx, maxx, miny, maxy, minz, maxz = bounding_box(mesh_leg[thigh])
    L_thigh = float(maxz - minz)
    pivot = [(minx + maxx) / 2.0, (miny + maxy) / 2.0, maxz]
    offset = [hip[0] - pivot[0], hip[1] - pivot[1], hip[2] - pivot[2]]
    # mesh_leg[thigh].x += offset[0]
    # mesh_leg[thigh].y += offset[1]
    # mesh_leg[thigh].z += offset[2]

    # Find Length of lower leg
    minx, maxx, miny, maxy, minz, maxz = bounding_box(mesh_leg[thigh])
    L_femer = float(maxz - minz)

    # Rotate Parts
    mesh_leg[thigh].rotate_using_matrix(thigh_RotM, hip)

    thigh_Vec = np.array([0, 0, -L_thigh])
    knee = np.dot(thigh_Vec, thigh_RotM)

    femer_Vec = np.array([0., 0., -L_femer])
    ankle = np.dot(femer_Vec, femer_RotM)

    # Move the femur to the knee location
    mesh_leg[lowerLeg].x += knee[0]  # + offset[0]
    mesh_leg[lowerLeg].y += knee[1]  # + offset[1]
    mesh_leg[lowerLeg].z += knee[2]  # + offset[2]

    # Rotate around point knee
    mesh_leg[lowerLeg].rotate_using_matrix(femer_RotM, knee)

    knee_Ang = angBetweenV(knee, ankle)

    hip_plane_v = np.array([knee[0], knee[1], 0])
    hip_Ang = angBetweenV(hip_plane_v, knee)

    mesh_leg[thigh].rotate(x_axis, radians(180))
    mesh_leg[lowerLeg].rotate(x_axis, radians(180))

    return mesh_leg, hip, knee, ankle, knee_Ang, hip_Ang


def generate_legmesh_YPR(thigh_ypr, femur_ypr, debug=False):
    # Leg pointers

    thigh = 0
    lowerLeg = 1

    # Mesh Alignment Constants
    hip = [0, 0, 0]

    mesh_leg = []
    # Load the STL files and add the vectors to the plot
    this_file_dir = Import_Utilis.get_this_dir(__file__)
    thigh_mesh_dir = this_file_dir + '\STL_Thigh.stl'
    femur_mesh_dir = this_file_dir + '\STL_LowerLeg.stl'

    # Add meshes to list
    mesh_leg.append(mesh.Mesh.from_file(thigh_mesh_dir))
    mesh_leg.append(mesh.Mesh.from_file(femur_mesh_dir))

    # Rotate Leg meshes to initial state
    mesh_leg[thigh].rotate(z_axis, radians(thigh_ypr[0]))
    mesh_leg[lowerLeg].rotate(z_axis, radians(femur_ypr[0]))

    # mesh_leg[thigh].rotate(x_axis, radians(180))
    # mesh_leg[lowerLeg].rotate(x_axis, radians(180))

    # mesh_leg[thigh].rotate(y_axis, radians(-90))
    # mesh_leg[lowerLeg].rotate(y_axis, radians(-90))

    # Center Hip
    minx, maxx, miny, maxy, minz, maxz = bounding_box(mesh_leg[thigh])
    L_thigh = float(maxz - minz)
    pivot = [(minx + maxx) / 2.0, (miny + maxy) / 2.0, maxz]
    offset = [hip[0] - pivot[0], hip[1] - pivot[1], hip[2] - pivot[2]]
    mesh_leg[thigh].x += offset[0]
    mesh_leg[thigh].y += offset[1]
    mesh_leg[thigh].z += offset[2]

    # Find Length of lower leg
    minx, maxx, miny, maxy, minz, maxz = bounding_box(mesh_leg[thigh])
    L_femer = float(maxz - minz)

    # Rotate Parts
    thigh_RotM = ypr2rotM(thigh_ypr)
    femur_RotM = ypr2rotM(femur_ypr)
    mesh_leg[thigh].rotate_using_matrix(thigh_RotM, hip)

    thigh_Vec = np.array([0, 0, -L_thigh])
    knee = np.dot(thigh_Vec, thigh_RotM)

    femer_Vec = np.array([0, 0, -L_femer])
    ankle = rotM_vec(femur_RotM, femer_Vec)  # np.dot(femer_RotM,femer_Vec)

    # Move the femur to the knee location
    mesh_leg[lowerLeg].x += knee[0]
    mesh_leg[lowerLeg].y += knee[1]
    mesh_leg[lowerLeg].z += knee[2]

    # Rotate around point knee
    # mesh_leg[lowerLeg].rotate_using_matrix(femer_RotM, knee)

    knee_Ang = angBetweenV(knee, ankle)

    hip_plane_v = np.array([knee[0], knee[1], 0])
    hip_Ang = angBetweenV(hip_plane_v, knee)

    return mesh_leg, hip, knee, ankle, knee_Ang, hip_Ang


if __name__ == "__main__":
    v1 = np.array([0, 0, 1])
    v2 = np.array([1, 0, 0])
    print(angBetweenV(v1, v2))
    # # Create a new plot
    # figure = pyplot.figure()
    # axes = mplot3d.Axes3D(figure)
    #
    # # Generate Leg mesh
    # thigh_Quat = [-0.03, -0.02, 0.7, 0.72]
    # femer_quat = [0.0, 0.0, 1.0, 0.0]
    #
    # thigh_RotV = Quat2RotM(thigh_Quat)
    # femer_RotV = Quat2RotM(femer_quat)
    #
    # print("Rotation Matrices: ", thigh_RotV, femer_RotV)
    #
    # # thigh_RotV = [45, 0, 0, 0]
    # # femer_RotV = [0, 0, 0, 0]
    # mesh_leg, hip, knee, ankle = generate_legmesh_Quat(thigh_RotV, femer_RotV)
    # # mesh_leg, hip, knee, ankle = generate_legmesh_Quat(thigh_Quat, femer_quat)
    #
    # for m in mesh_leg:
    #     axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
    #
    # x = [hip[0], knee[0]]
    # y = [hip[1], knee[1]]
    # z = [hip[2], knee[2]]
    #
    # axes.plot(x, y, z, c='r', marker='o')
    #
    # generateSensor = False
    # if generateSensor == True:
    #     coverage = 0.4
    #     vector = np.array(hip) - np.array(knee)
    #     X, Y, Z = generate_sensor_mesh(np.array(knee) + vector * coverage, knee)
    #     axes.plot_surface(X, Y, Z, color='black', alpha=0.5)
    #
    #     vector = np.array(knee) - np.array(ankle)
    #     X, Y, Z = generate_sensor_mesh(knee, np.array(knee) - vector * coverage)
    #     axes.plot_surface(X, Y, Z, color='black', alpha=0.5)
    #
    # # Auto scale to the mesh size
    # scale = np.concatenate([m.points for m in mesh_leg]).flatten("C")
    # axes.auto_scale_xyz(scale, scale, scale)
    #
    # # Configure Axes
    # pyplot.xlim(-1, 1)
    # pyplot.ylim(-1, 1)
    # # pyplot.set_zlim(0, 10)
    # pyplot.xlabel('Coronal Plane')
    # pyplot.ylabel('Sagittal Plane')
    #
    # # Show the plot to the screen
    # pyplot.show()

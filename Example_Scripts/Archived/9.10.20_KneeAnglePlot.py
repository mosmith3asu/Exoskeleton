from Modules.Plotting import RenderMesh, Simple_Plot
import numpy as np
from numpy import savetxt
import time
from numpy import genfromtxt
#NEED TO REINITIALIZE FOR EVERY PASS
imu_path = 'C:\\Users\\mason\\Desktop\\Patient_files\\Patient3S3_fixed.csv'


    # Import data as array
data = genfromtxt(imu_path, delimiter=',')


    # remove header row from import data
patient = data[1:, :]

if __name__=="__main__":
    index_ref = 200
    patient_num = 1

    data = np.array([[0,0]])
    quat_thigh = patient[index_ref, 0:4]
    quat_femur = patient[index_ref, 4:8]

    rotM_thigh = RenderMesh.quat2rotM(quat_thigh)
    rotM_femur = RenderMesh.quat2rotM(quat_femur)

    init_rotM_thigh = RenderMesh.inv_rotM(rotM_thigh)
    init_rotM_femur =  RenderMesh.inv_rotM(rotM_femur)

    init_angle = RenderMesh.angBetweenRotM(init_rotM_thigh, init_rotM_femur)

    print("verifying inverse returns identity matrix")
    print(np.dot(init_rotM_thigh,rotM_thigh).astype(np.int))
    print(np.dot(init_rotM_femur,rotM_femur).astype(np.int))

    time.sleep(1)

    #intit_angle = RenderMesh.angBetweenRotM(init_rotM_thigh,init_rotM_femur)

    for index in range(len(patient[:,0])):
    #for index in range(10000):
        quat_thigh = patient[index, 0:4]
        quat_femur = patient[index, 4:8]

        rotM_thigh = RenderMesh.quat2rotM(quat_thigh)
        rotM_femur = RenderMesh.quat2rotM(quat_femur)

        rotM_thigh = np.dot(rotM_thigh,init_rotM_thigh)
        rotM_femur = np.dot(rotM_femur,init_rotM_femur)

        angle = RenderMesh.angBetweenRotM(rotM_thigh,rotM_femur) - 20
        data_new = np.array([[index,angle]])
        data = np.append(data,data_new,axis=0)
        print(f'progress: {index}/{len(patient[:,0])}')

    savetxt(f'data_kneeAng_patient{patient_num}.csv', data, delimiter=',')
    plt,fig,ax = Simple_Plot.plot2D(data,f'Knee Angle Patient {patient_num}, \nRef_i = {index_ref}')
    plt.show()

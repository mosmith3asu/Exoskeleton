from pathlib import Path
import os
import serial
from time import sleep
def add_new_dir(new_dir,parent_dir='C:\\Users\\mason\\Desktop\\Thesis\Patient_files\\Outputs'):
    from datetime import datetime
    time_begin = datetime.now()  # current date and time
    date = time_begin.strftime("%m_%d_%Y")
    directory=date+"_"+new_dir
    path = os.path.join(parent_dir, directory)
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    return path

def add_show_fig(copy2path):
    import shutil
    file_path = "C:\\Users\\mason\\Desktop\\Thesis\\ML_MasonSmithGit\Modules\\Utils\\show_figs_python3_7.py"
    newPath = shutil.copy(file_path, copy2path)

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_dir_in_proj(file_proj_path):
    proj_root_dir = get_project_root()
    dir = str(proj_root_dir) + file_proj_path
    return dir

def get_this_dir(os_file):
    return str(os.path.dirname(os_file))

# def serial_quat(COM,baud = 115200, backup_quat = [1,0,0,0]):
#     ser = serial.Serial('COM4', baud)
#     ser.open()
#     b = ser.readline()
#     str_rn = b.decode()
#     str = str_rn.rstrip()
#     return str
class serial_legsensor():
    def __init__(self, COM,baud = 115200):
        self.ser = serial.Serial(COM, baud)
        self.ser.close()
        self.ser.open()
        # q1 = Quaternion()
        # q2 = Quaternion([])
        self.backup_data = [10, 0, 0, 0, 0, 0,   # first sensor
        0, 0, 0, 0, 0, 0,   # first sensor
        0, 0, 0, 0, 0,10,   # first sensor
        0, 0, 0, 0, 0,      # second sensor
        0, 0, 0, 0, 0,      # second sensor
        ]

        print(f"Serial initialized {COM}")

    def read_line(self):
        #self.ser.reset_input_buffer()
        #self.ser.flushInput()
        #while ser.inWaiting()

        try:
            bytes = self.ser.read_all()
            bytes_str = bytes.decode()
            bytes_str_lst = bytes_str.split("\r")
            b_str = bytes_str_lst[-2]
            list_str = b_str.split(",")
            list_float = [float(item) for item in list_str]
            if all(v == 0 for v in list_float):
                list_float = self.backup_data
            self.backup_data = list_float
        except:
            list_float = self.backup_data


        return list_float

    def close_serial(self):
        self.ser.close()

class serial_quat():
    def __init__(self, COM,baud = 115200):
        self.ser = serial.Serial(COM, baud)
        self.ser.close()
        self.ser.open()
        # q1 = Quaternion()
        # q2 = Quaternion([])
        self.backup_quat = [1.,0.,0.,0.,1.,0.,0.,0.]
        print(f"serial initialized {COM}")

    def read_line(self):
        #self.ser.reset_input_buffer()
        #self.ser.flushInput()
        #while ser.inWaiting()

        try:
            bytes = self.ser.read_all()
            bytes_str = bytes.decode()
            bytes_str_lst = bytes_str.split("\r")
            b_str = bytes_str_lst[-2]
            list_str = b_str.split(",")
            list_float = [float(item) for item in list_str]
            if all(v == 0 for v in list_float):
                list_float = self.backup_quat
            self.backup_quat = list_float
        except:
            list_float = self.backup_quat


        return list_float

    def close_serial(self):
        self.ser.close()

if __name__=="__main__":
    ser = serial_quat("COM9")
    while True:
        print(ser.read_line())
        sleep(0.1)

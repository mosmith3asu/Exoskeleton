from PyQt5 import QtWidgets
from Modules.Interface.Data_Validation import Qt_Backend, QtDes_DataValidationInterface
import sys
if __name__ == "__main__":
    # Initialize Qt GUI
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = QtDes_DataValidationInterface.Ui_MainWindow()
    ui.setupUi(MainWindow)

    # Add Qt.Thread Member to GUI and Start
    thread = Qt_Backend.Qt_ThreadClass(ui)
    thread.ui = ui
    ui.thread = thread
    ui.thread.start()

    #Execute App
    MainWindow.show()
    app.exec_()


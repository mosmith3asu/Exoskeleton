from PyQt5 import QtWidgets
from Modules.Interface.Learning_GUI import Qt_Interface
import sys

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Qt_Interface.Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()

sys.exit(app.exec_())

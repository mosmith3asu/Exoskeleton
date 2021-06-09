if __name__ == "__main__":
    from PyQt5 import QtWidgets
    from Modules.Interface import Qt_Interface
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Qt_Interface.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())




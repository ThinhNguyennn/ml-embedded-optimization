import sys
import os
import numpy as np
import cv2 
import psutil
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow
from modelAPP import Ui_MainWindow
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt, QTimer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
import sysv_ipc
from multiprocessing import Value, Lock
from CheckResource import cpu_usage, ram_usage
from model_CNN import predict_label_with_CNN
from model_LSVM import predict_label_with_LSVM
from model_BNN import predict_label_with_BNN
from CheckCamrera import get_available_cameras

send_key = 12345  
receive_key = 56789                
message_queue = sysv_ipc.MessageQueue(send_key, sysv_ipc.IPC_CREAT)     

Page_message = ''                 
message_flag = Value('b', False)  
message_lock = Lock() 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
    
        self.uic.Start_Button.clicked.connect(self.start_capture_video)
        self.uic.Stop_Button.clicked.connect(self.stop_all_operations)
        self.uic.Predict_Button.clicked.connect(self.setup_predict_timer)
        
        self.thread = {}
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_CPU)
        self.timer.start(500) 
        
        self.uic.Model_Box.addItem("CNN")
        self.uic.Model_Box.addItem("BNN")
        self.uic.Model_Box.addItem("LSVM")
        
        self.predict_timer = QTimer(self)  
        self.predict_timer.timeout.connect(self.start_prediction)
        
        self.receiver_thread = ReceiverThread(receive_key, message_flag, message_lock)
        self.receiver_thread.message_received.connect(self.update_model_name)
        self.receiver_thread.start()
        
        self.model_name = "CNN"  
        
        self.sender_thread = SenderThread(send_key)
        self.uic.Model_Box.currentTextChanged.connect(self.send_model_name)
        
        self.camera_list()
        
    def camera_list(self):
        available_cameras = get_available_cameras()
        for device_index, device_name in available_cameras.items():
            self.uic.Camera_Box.addItem(device_name)
    
    def stop_all_operations(self):
        self.stop_capture_video()
        self.stop_prediction()
        
    def closeEvent(self, event):
        self.stop_capture_video()
        self.stop_prediction()
        event.accept()
        
    def stop_capture_video(self):
        if 1 in self.thread:
            self.thread[1].stop()
            self.thread[1].wait()           
            del self.thread[1]
    
    def stop_and_remove_thread(self):
        if 1 in self.thread:
            self.thread[1].stop()
            self.thread[1].wait()
            del self.thread[1]
                
    def start_capture_video(self):
        selected_index = self.uic.Camera_Box.currentIndex()
        if selected_index != -1:
            selected_device_index = list(get_available_cameras().keys())[selected_index]
            self.stop_and_remove_thread()
            self.thread[1] = capture_video(index=1, device_index=selected_device_index)
        else:
            self.stop_and_remove_thread() 
            self.thread[1] = capture_video(index=1)  
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_webcam)
    
    def show_webcam(self, cv_img):
        self.cv_img = cv_img
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.Camera_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(400, 400, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
        
    def check_CPU(self):
        cpu_percent = int(cpu_usage(psutil.cpu_percent()) * 100)
        ram_percent = int(ram_usage(psutil.virtual_memory().percent) * 100)
        self.uic.CPU_label.setText(f"{cpu_percent}%")
        self.uic.RAM_label.setText(f"{ram_percent}%")

    def start_prediction(self):
        # model_name = self.uic.Model_Box.currentText()
        # self.model_name = model_name
        self.thread[2] = PredictThread(index = 2,cv_img=self.cv_img, model_name=self.model_name)
        self.thread[2].start()
        self.thread[2].signal.connect(self.display_prediction)
    
    def setup_predict_timer(self):
        self.predict_timer.start(700)
        
    def stop_prediction(self):
        if 2 in self.thread:
            self.thread[2].quit()
            self.thread[2].wait()
            del self.thread[2]
            
    def display_prediction(self, result):
        self.uic.Predict_label.setText(result)
        
    def update_model_name(self, message):
        cleaned_message = message.rstrip('\x00')
        self.model_name = cleaned_message
        self.uic.Model_label.setText(f"Model:{self.model_name}")
        print("receive_message",message)
        
    def send_model_name(self, model_name):
        self.sender_thread.send_message(model_name)
        
class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    
    def __init__(self, index,device_index):
        super().__init__()
        self.index = index
        self.device_index = device_index
        self.running = True  # Add a running flag
        print("start threading", self.index)
    
    def run(self):
        cap = cv2.VideoCapture(self.device_index)
        while self.running:  # Use the running flag to control the loop
            ret, cv_img = cap.read()
            if ret:
                self.signal.emit(cv_img)
        cap.release()  # Release the video capture resource when done
        print("Video capture released.")
    
    def stop(self):
        print("stop threading", self.index)
        self.running = False  # Set the running flag to False
        self.wait()  # Wait for the thread to finish

class PredictThread(QThread):
    signal = pyqtSignal(str) 
    
    def __init__(self, index, cv_img=None, model_name=None):
        super().__init__()
        self.index = index 
        self.cv_img = cv_img
        self.model_name = model_name
        self.running = True
        print("start threading", self.index)
    
    def run(self):
        if self.running:
            # Convert the OpenCV image to the format expected by the prediction functions
            img_rgb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
            #pil_img = Image.fromarray(img)
            resized_img = cv2.resize(img_rgb, (28, 28))
            # Perform prediction based on selected model
            if self.model_name == "CNN":
                label = str(predict_label_with_CNN(resized_img))
            elif self.model_name == "BNN":
                label = str(predict_label_with_BNN(resized_img))
            elif self.model_name == "LSVM":
                label = str(predict_label_with_LSVM(resized_img))
            else:
                print("Khong nhan dien:", self.model_name)
                label = "Unknown Model"        
            self.signal.emit(label)
        else:
            print("thread 2 is not running")
        
    def stop(self):
        print("stop threading", self.index)
        self.running = False
        self.wait() 

class ReceiverThread(QThread):
    message_received = pyqtSignal(str)      

    def __init__(self, receive_key, flag, lock, parent=None):
        super().__init__(parent)
        self.message_queue = sysv_ipc.MessageQueue(receive_key, sysv_ipc.IPC_CREAT) 
        self.flag = flag        
        self.lock = lock       

    def run(self):
        while True:
            try:
                message, _ = self.message_queue.receive()
                decoded_message = message.decode('utf-8')
                cleaned_message = decoded_message.strip()
                self.message_received.emit(cleaned_message)
                print("receive_message",cleaned_message)
                with self.lock:
                    self.flag.value = True
            except sysv_ipc.ExistentialError:           
                print(f"Message Queue with key {self.message_queue.key} does not exist.")

class SenderThread(QThread):
    def __init__(self, send_key, parent=None):
        super().__init__(parent)
        self.message_queue = sysv_ipc.MessageQueue(send_key, sysv_ipc.IPC_CREAT)

    def send_message(self, message):
        self.message = message
        if not self.isRunning():
            self.start()

    def run(self):
        encoded_message = self.message.encode('utf-8')
        self.message_queue.send(encoded_message)
        print(f"Message sent: {self.message}")
                            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
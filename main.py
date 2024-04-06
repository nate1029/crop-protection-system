import tkinter as tk
import cv2

def CreateDict():
    dict = {}
    f = open('dnn_model/classes.txt', 'r')
    ct = 0
    while True:
        line = f.readline()
        if not line:
            break
        else:
            dict[ct] = line.strip()
            ct = ct + 1
    return dict

def RetString(dict, id):
    return dict[id]

class Frames():
    def __init__(self):
        dict = CreateDict()
        net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")  # model deja antrenat
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(320, 320), scale=1/255)
        capture = cv2.VideoCapture(0)
        cv2.namedWindow("Frame", cv2.WINDOW_GUI_NORMAL)
        # Use WINDOW_NORMAL flag for resizable window
        while True:
            ret, frame = capture.read()

            (class_ids, scores, bboxes) = model.detect(frame)
            print("class_id: ", class_ids, "\nscore: ", scores, " boxes: ", bboxes, "\n")

            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                (x, y, w, h) = bbox
                cv2.putText(frame, RetString(dict, class_id), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 0, 50), 2)

            cv2.imshow("Frame ", frame)
            cv2.waitKey(1)

class MainWindow():
    def __init__(self, name):
        window = tk.Tk()
        window.title(name)
        window.resizable(False, False)
        window.geometry('400x100')
        window['bg'] = 'red'
        buttonsFrame = tk.Frame(master=window, background='red')
        startButton = tk.Button(master=buttonsFrame, text="Start Detection!", width=13, command=self.start_detection)
        buttonsFrame.pack(padx=10)
        startButton.pack(padx=10, pady=10)
        window.mainloop()

    def start_detection(self):
        Frames()

if __name__ == '__main__':
    win = MainWindow("Object Detection")
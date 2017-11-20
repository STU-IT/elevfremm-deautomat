from tkinter import *
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
import numpy as np
#import PIL
#from PIL import Image, ImageTk
from PIL import Image, ImageTk

def getSub():
    sub = open("subjects.csv")
    subjects = sub.read().split(",")
    subjects[len(subjects)- 1] = subjects[len(subjects) - 1].strip("")
    sub.close()
    
    if(len(subjects) == 1):
        subjects = [""]
    else:
        tempSubjects = [""]
        for i in range(len(subjects) - 1):
            #print(i)
            tempSubjects.append(subjects[i])
        subjects = tempSubjects
    #print(subjects)
    return subjects
    
def takeImg(personLabel = 0):
    global dirs
    global window
    clearGUI()
    count = 0

    imgLabel = Label(window).pack()
    
    for widget in window.winfo_children():
        widget.destroy()
    #window.mainloop()
    
    if(personLabel == 0):
        if(len(dirs) == 0):
            os.makedirs("person/s1")
        else:
            os.makedirs("person/s" + str(len(dirs) + 1))
                
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
        image = frame.array
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        testIma = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30)
        )

        for(x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
        # show the frame
        #cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        
        #imgLabel.configure()

        #current_Ima = Image.fromarray(gray)
        #imgtk = ImageTk.PhotoImage(image = gray)
        img = ImageTk.PhotoImage(testIma)
        Label(window, image = img).pack()
        
 
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            #print(dirs)
            dirs = os.listdir("person")
            #print(dirs)
            if(personLabel == 0):
                folder = "s" + str(len(dirs))
                folder = dirs.index(folder)
                #print(folder)
                #images = os.listdir("person/" + dirs[len(dirs) - 1])
                #camera.capture("person/" + dirs[len(dirs) - 1] + "/" + str(len(images) + 1) + ".jpg")
                images = os.listdir("person/" + dirs[folder])
                camera.capture("person/" + dirs[folder] + "/" + str(len(images) + 1) + ".jpg")

            else:
                images = os.listdir("person/" + dirs[personLabel - 1])
                camera.capture("person/" + dirs[personLabel - 1] + "/" + str(len(images) + 1) + ".jpg")
            count = count + 1
            
            if(count == 5):
                break
            else:
                continue
    def finsh():
        global subjects
        #print(subjects)
        
        sub = open("subjects.csv", "a")#append and read
        sub.write(tbox.get() + ",")
        sub.close()
        
        subjects = getSub()
        #print(subjects)
                
        sub.close()
        app.destroy()
            
    cv2.destroyAllWindows()
    if(personLabel == 0):
        app = App("tjek ind V2")
        Text(app, "please enter your name")
        tbox = TextBox(app)
        PushButton(app, finsh, text="Save name")
        app.display()

def trainer():
    def detect_face(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30)
        )
    
        if(len(faces) == 0):
            return None, None

        (x, y, w, h) = faces[0]

        return gray[y:y+w, x:x+h], faces[0]

    def prepare_training_data(data_folder_path):
        dirs = os.listdir(data_folder_path)

        faces = []

        labels = []

        #folder starts with 's'
        for dir_name in dirs:
            if(not dir_name.startswith("s")):
                continue

            #remove the 's' give us the lable
            label = int(dir_name.replace("s", ""))

            subject_dir_path = data_folder_path + "/" + dir_name

            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:
                if(image_name.startswith(".")):
                    continue

                image_path = subject_dir_path + "/" + image_name

                #print(image_name)
            
                image = cv2.imread(image_path)

                face, rect = detect_face(image)

                if(face is not None):
                    faces.append(face)
                    labels.append(label)

        return faces, labels

    print("Preparing data...")
    faces, labels = prepare_training_data("person")
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write("recData.xml")

def recognition():
    global subjects
    trainer()
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("recData.xml")
    
    count = 0
    isYouCount = 0

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30, 30)
        )
    
        #print(faces)
        if(faces == ()):
            count = 0
            isYouCount = 0
        for(x, y, w, h) in faces:
            face = gray[y:y+w, x:x+h]

            label = face_recognizer.predict(face)
            #print(label)

            label_text = subjects[label[0]]

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            if(label[1] < 110):
                isYouCount = isYouCount + 1
                if(isYouCount == 3):
                    cv2.putText(image, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.destroyAllWindows()
                    rawCapture.truncate(0)
                    return label[0]
            else:
                #if 5 sec or more is unknown. ask you to registrant
                isYouCount = 0
                count = count + 1
                #print(count)
                if(count >= 20):
                    cv2.destroyAllWindows()
                    rawCapture.truncate(0)
                    return "Unknown"
                cv2.putText(image, "Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
 
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    
def welcome():
    global window
    global name

    clearGUI()

    Label(window, text="Welcome " + name, font=(None, 30), background='white').place(relx=0.5, rely=0.05, anchor=N)
    window.after(3000, mainGUI)
    
def regBefore():
    global window
    
    clearGUI()

    '''
    Label(window, text="have registratet before?", font=(None, 15)).grid(row=0, column=0, columnspan=3)
    Button(window, text = "no", command = willYouReg, width=50, height=20).grid(row=1, column=0)
    Button(window, text = "yes", command = haveRegBefore, width=50, height= 20).grid(row=1, column=1)

    '''

    Label(window, text="have registratet before?", font=(None, 30), background='white').place(relx=0.5, rely=0.05, anchor=N)
    Button(window, text = "no", font=(None, 25), command = willYouReg, width=25, height=10).place(relx=0.3, rely=0.2, anchor=N)
    Button(window, text = "yes", font=(None, 25), command = haveRegBefore, width=25, height= 10).place(relx=0.7, rely=0.2, anchor=N)
    
def willYouReg():
    global window

    clearGUI()
    Label(window, text="Will you registrate?", font=(None, 30, ), background='white').place(relx=0.5, rely=0.05, anchor=N)
    Button(window, text="no", font=(None, 25), command=mainGUI, width=25, height=10).place(relx=0.3, rely=0.2, anchor=N)
    Button(window, text="yes", font=(None, 25), command=takeImg, width=25, height=10).place(relx=0.7, rely=0.2, anchor=N)

def haveRegBefore():
    global window
    global subjects

    #print(subjects)
    
    nameRow = 0
    count = 0
    
    clearGUI()

    Label(window, text = "click your name")
    nameFrame = Frame(window)
    nameFrame.grid_rowconfigure(5, weight=1)
    nameFrame.grid_columnconfigure(5, weight=1)
    
    for i in range(1, len(subjects)):
        if((i - 1) % 5 == 0):
            nameRow = nameRow + 1
            count = 0
        count = count + 1

        Button(nameFrame, command=lambda: takeImg([i]), text=subjects[i]).grid(row=nameRow, column=count - 1)

    nameFrame.pack()

def clearGUI():
    global window

    for widget in window.winfo_children():
        widget.destroy()
        
def mainGUI():
    global name
    global ismainloob
    
    clearGUI()        
    
    if(len(subjects) == 1):
        Label(window, text="be the first", font=(None, 30), background='white').place(relx=0.5, rely=0.05, anchor=N)
        Button(window, text = "Take picture", font=(None, 25), command = takeImg, width=25, height=10).place(relx=0.5, rely=0.2, anchor=N)
        #window.mainloop()
    else:
        rec = recognition()
        if(rec == "Unknown"):
            regBefore()
        else:
            name = subjects[rec]
            
            Label(window, text="Are you " + name, font=(None, 30), background='white').place(relx=0.5, rely=0.05, anchor=N)
            Button(window, text = "no", font=(None, 25), command = regBefore, width=25, height=10).place(relx=0.3, rely=0.2, anchor=N)
            Button(window, text = "yes", font=(None, 25), command = welcome, width=25, height=10).place(relx=0.7, rely=0.2, anchor=N)
    
#-------------------------------------------------
#code stuff

dirs = os.listdir("person")
subjects = getSub()

cascPath = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

camera = PiCamera()
camera.resolution = (640, 480) #be Full screen
camera.framerate = 30

rawCapture = PiRGBArray(camera, size=(640, 480)) #be Full screen

#column x
#row y
window = Tk()
window.geometry("1920x1080")#1920x1080

window.configure(background='white')

window.grid_rowconfigure(2, weight=1)
window.grid_columnconfigure(2, weight=1)

#window.overrideredirect(1)
takeImg()

mainGUI()

window.mainloop()

'''
window = Tk()

def close_window():
    for widget in window.winfo_children():
        widget.destroy()
        #window.pack_forget()
        button = Button (window, text = "hej.", command = window.destroy)
        button.pack()

window.geometry("500x500")
window.overrideredirect(1)
button = Button (window, text = "Good-bye.", command = close_window)
button.pack()

window.mainloop()
'''

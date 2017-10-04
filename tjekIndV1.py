from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
import numpy as np
from guizero import App, Text, PushButton, TextBox, Box 

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
            print(i)
            tempSubjects.append(subjects[i])
        subjects = tempSubjects
    print(subjects)
    return subjects
    '''
    print(len(subjects))
    print(subjects)
    '''
    
def takeImg():
    global dirs
    global app
    count = 0
    app.destroy()
    
    if(len(dirs) == 0):
        os.makedirs("person/s1")
    else:
        os.makedirs("person/s" + str(len(dirs) + 1))
                
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
        image = frame.array
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30)
        )

        for(x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
 
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                
            dirs = os.listdir("person")

            images = os.listdir("person/" + dirs[len(dirs) - 1])
            camera.capture("person/" + dirs[len(dirs) - 1] + "/" + str(len(images) + 1) + ".jpg")

            count = count + 1
            
            if(count == 5):
                break
            else:
                continue
    def finsh():
        global subjects
        print(subjects)
        
        sub = open("subjects.csv", "a")#append and read
        sub.write(tbox.get() + ",")
        sub.close()
        
        subjects = getSub()
        print(subjects)
                
        sub.close()
        app.destroy()
            
    cv2.destroyAllWindows()
    
    app = App("tjek ind V1")
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
            minSize = (30, 30),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
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

                print(image_name)
            
                image = cv2.imread(image_path)

                #view image
                #cv2.imshow("Training in image...", image)
                #cv2.waitKey(100)

                face, rect = detect_face(image)

                if(face is not None):
                    faces.append(face)
                    labels.append(label)

                #cv2.destroyAllWindows()
                #cv2.waitKey(1)
                #cv2.destroyAllWindows()
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
    
        print(faces)
        if(faces == ()):
            count = 0
        for(x, y, w, h) in faces:
            face = gray[y:y+w, x:x+h]

            label = face_recognizer.predict(face)
            print(label)

            label_text = subjects[label[0]]

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            if(label[1] < 110):
                cv2.putText(image, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.destroyAllWindows()
                return label[0]
            else:
                #if 5 sec or more is unknown. ask you to registrant
                count = count + 1
                print(count)
                if(count >= 20):
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

def noToRegBefore():
    print("")
    
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


app = App("tjek ind V1")
if(len(subjects) == 1):
    Text(app, "be the first")
    PushButton(app, takeImg ,text="Take picture")
    app.display()
else:
    #check if face is known
    print("")
rec = recognition()
if(rec == "Unknown"):
    Text(app, "have registratet before?")
    box = Box(app, layout="grid")
    PushButton(box, takeImg ,text="yes", grid=[0,1])
    PushButton(box, takeImg ,text="no", grid=[0,0])
    app.display()
else:
    name = subjects[rec]

    Text(app, "are you " + name)
    app.display()

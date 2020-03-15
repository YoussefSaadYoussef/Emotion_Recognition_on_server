import base64
from io import BytesIO
from PIL import Image
import cv2
from gtts import gTTS
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import imutils


def encode(img):
    encoded_string = base64.b64encode(img)
    return encoded_string


def decode(base64_string):
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    image = image.convert("RGB")

    return np.array(image)




def emotion_finder(image_path):
    
    # parameters for loading data and images
    detection_model_path = 'haarcascade_frontalface_default.xml'
    emotion_model_path = 'gpu_mini_XCEPTION.63-0.64.hdf5'
    
    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
     "neutral"]

    img = cv2.imread(image_path)
    
    img = imutils.resize(img,width=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #to get faces from image 
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    
    imgClone = img.copy()
    
    # to classifiy image and get the prediction in label 
    
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
    
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
            
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
        cv2.putText(imgClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(imgClone, (fX, fY), (fX + fW, fY + fH),
                                 (0, 0, 255), 2)
    
    return imgClone, [label]

def read_emotion(emotion):
    mytext = ""
    for i in range(len(emotion)):
        mytext += emotion[i] + " "
        if i < len(emotion)-1:
            mytext += "and "

    myobj = gTTS(text=mytext, lang='en', slow=False)
    myobj.save("output.mp3")
    sound_bytes = open("output.mp3", "rb").read()

    return base64.b64encode(sound_bytes)
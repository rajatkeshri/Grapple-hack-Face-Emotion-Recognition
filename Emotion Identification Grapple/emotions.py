import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import pandas as pd


USE_WEBCAM = False # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []

writer = pd.ExcelWriter('expression_identification_train.xlsx', engine='xlsxwriter')

for i in range(1,2):
    ###############################################
    frame_number = []
    no_of_faces = []
    bounding_box = []
    expression = []
    accuracy = []

    frame = 1
    ###########################################
    # Select video or webcam feed
    cap = None
    if (USE_WEBCAM == True):
        cap = cv2.VideoCapture(0) # Webcam source
    else:
        #cap = cv2.VideoCapture('./demo/video'+str(i)+'_test.mp4') # Video file source
        cap = cv2.VideoCapture('./demo/video2_test.mp4') # Video file source

    while cap.isOpened(): # True:
        ret, bgr_image = cap.read()
        #bgr_image = video_capture.read()[1]

        if ret == True:

            #bgr_image = cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE)
            bgr_image = cv2.resize(bgr_image,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
        			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                    print(emotion_text,emotion_probability)
                    expression.append(2)
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                    print(emotion_text,emotion_probability)
                    expression.append(4)
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                    print(emotion_text,emotion_probability)
                    expression.append(1)
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                    print(emotion_text,emotion_probability)
                    expression.append(5)
                else:
                    color = emotion_probability * np.asarray((0, 0, 0))
                    print(emotion_text,emotion_probability)
                    expression.append(3)

                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

                ##############################################
                frame_number.append(frame)
                no_of_faces.append(len(faces))
                #bounding_box.append(face_coordinates)
                bounding_box.append([x1, y1, x2, y2])
                accuracy.append(emotion_probability*100)
                ##############################################

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('window_frame', bgr_image)
            frame+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    dict = {'Frame Number': frame_number,
            'Number of detected Faces': no_of_faces,
            'Bounding box': bounding_box,
            'Expression' : expression,
            'Accuracy' : accuracy}
    print(len(frame_number),len(no_of_faces),len(bounding_box),len(expression),len(accuracy))
    df = pd.DataFrame(dict)
    print(df)
    df.to_excel(writer, sheet_name='video'+str(i))

    cap.release()
    #cv2.destroyAllWindows()
writer.save()

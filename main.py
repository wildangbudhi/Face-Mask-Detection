import cv2
import numpy as np
import tensorflow as tf

print( "Load Model" )

FACE_DETECTOR = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
MASK_CLASSIFIER = tf.keras.models.load_model('./model/masknet.h5')

mask_dict = { 
    0: {
        "label" : 'MASK',
        "color" : (0,255,0)
    }, 
    1: {
        "label" : 'NO MASK',
        "color" : (255,0,0)
    }
}

print( "Init Camera" )
  
vid = cv2.VideoCapture(1)
  
while(True):
      
    ret, frame = vid.read()

    img = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    faces = FACE_DETECTOR.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4)

    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        crop = new_img[y:y+h,x:x+w]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = MASK_CLASSIFIER.predict(crop)
        mask_class = mask_result.argmax()
        cv2.putText( new_img, mask_dict[ mask_class ]["label"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, mask_dict[ mask_class ]["color"] )
        cv2.rectangle( new_img, (x,y), (x+w,y+h), mask_dict[ mask_class ]["color"])

    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', new_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import face_recognition
import time


face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

source=cv2.VideoCapture(0)

color_dict={0:(0,255,0),1:(0,0,255)}

amit_image = face_recognition.load_image_file('images/samples/amit.jpg')
amit_face_encodings = face_recognition.face_encodings(amit_image)[0]

ignatius_image = face_recognition.load_image_file('images/samples/ignatius.jpg')
ignatius_face_encodings = face_recognition.face_encodings(ignatius_image)[0]

akshay_image = face_recognition.load_image_file('images/samples/akshay.jpg')
akshay_face_encodings = face_recognition.face_encodings(akshay_image)[0]

devi_image = face_recognition.load_image_file('images/samples/devi.jpg')
devi_face_encodings = face_recognition.face_encodings(devi_image)[0]

known_face_encodings = [amit_face_encodings,ignatius_face_encodings,akshay_face_encodings,devi_face_encodings]

known_face_names = ["Amit","Ignatius","Akshay","Devi"]

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)
    name_of_person = "Unknown"

    for x,y,w,h in faces:  
        if(w > 150):
            index=0
            face_img=gray[y:y+w,x:x+w]
            start = time.time()
            all_face_locations = face_recognition.face_locations(img,model='hog')
            all_face_encodings = face_recognition.face_encodings(img,all_face_locations)         
            for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
                all_matches = face_recognition.compare_faces(known_face_encodings,current_face_encoding,tolerance=0.5)
                if True in all_matches:
                    first_match_index = all_matches.index(True)
                    name_of_person = known_face_names[first_match_index]
            end = time.time()
            print(f"Time taken to compare {name_of_person}'s face : {end - start}sec")        
            
        else:
            index=1
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[index],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[index],-1)
        cv2.putText(
          img, name_of_person, 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
source.release()
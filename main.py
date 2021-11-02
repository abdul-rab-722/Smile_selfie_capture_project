import cv2

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("D:\Projects\OpenCV Projects\Smile-selfie-capture-project\dataset\haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("D:\Projects\OpenCV Projects\Smile-selfie-capture-project\dataset\haarcascade_smile.xml")

while True:
    success,img = video.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg,1.1,3)
    cnt=500
    keyPressed = cv2.waitKey(1)
    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),4)
        smiles = smileCascade.detectMultiScale(grayImg,1.8,15)
        for x,y,w,h in smiles:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),5)
            print("Image "+str(cnt)+"Saved")
            path=r'D:\Projects\OpenCV Projects\Smile-selfie-capture-project\images\img'+str(cnt)+'.jpg'
            cv2.imwrite(path,img)
            cnt +=1
            if(cnt>=503):   
                break
                
    cv2.imshow('live video',img)
    if(keyPressed & 0xFF==ord('q')):
        break

video.release()                                  
cv2.destroyAllWindows() 

from django.shortcuts import render, redirect
from . models import UserPersonalModel
from . forms import UserPersonalForm, UserRegisterForm
from django.contrib.auth import authenticate, login,logout
from django.contrib import messages
import numpy as np

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


def Landing_1(request):
    return render(request, '1_Landing.html')

def Register_2(request):
    form = UserRegisterForm()
    if request.method =='POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(request, 'Account was successfully created. ' + user)
            return redirect('Login_3')

    context = {'form':form}
    return render(request, '2_Register.html', context)


def Login_3(request):
    if request.method =='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('Home_4')
        else:
            messages.info(request, 'Username OR Password incorrect')

    context = {}
    return render(request,'3_Login.html', context)

def Home_4(request):
    return render(request, '4_Home.html')

def Teamates_5(request):
    return render(request,'5_Teamates.html')

def Domain_Result_6(request):
    return render(request,'6_Domain_Result.html')

def Problem_Statement_7(request):
    return render(request,'7_Problem_Statement.html')
    

def Per_Info_8(request):
    if request.method == 'POST':
        fieldss = ['firstname','lastname','age','address','phone','city','state','country']
        form = UserPersonalForm(request.POST)
        if form.is_valid():
            print('Saving data in Form')
            form.save()
        return render(request, '4_Home.html', {'form':form})
    else:
        print('Else working')
        form = UserPersonalForm(request.POST)    
        return render(request, '8_Per_Info.html', {'form':form})
    

def Deploy_9(request):
    if request.method == "POST":
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        classifier = Classifier("C:/Users/Imran/Music/MAIN_PROJECT/CODE/DEPLOYMENT/PROJECT/APP/keras_model.h5")
        
        offset = 20
        imgSize = 300
        
        labels = ["Callme","Ok","Rock","Thumbs Down", "Thumbs Up"]

        s1 = 0

        while True:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            
            try:      
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        print(prediction, index)
                        if index == 0:          
                            s1 += 1
                            if s1 >= 20:
                                print("HI DETECTED")
                                s1 = 0

                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)

                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)

                    cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                                (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x-offset, y-offset),
                                (x + w+offset, y + h+offset), (255, 0, 255), 4)
                    
        
            except:
                pass
                
            cv2.imshow("Image", imgOutput)
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            

        cv2.destroyAllWindows()
        return render (request, '9_Deploy.html')
    else:
        return render (request, '9_Deploy.html')


def Per_Database_10(request):
    models = UserPersonalModel.objects.all()
    return render(request, '10_Per_Database.html', {'models':models})


def Logout(request):
    logout(request)
    return redirect('Landing_1')

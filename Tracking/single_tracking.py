import cv2
import sys
from random import randint

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[5]
#print(tracker_type)

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()
#aici am creat fiecare obiect pentru algortimi. 
#print(tracker)

video = cv2.VideoCapture('Videos/racee.mp4') #este un opencv class ca sa deschidem video ul 
if not video.isOpened():#verificam daca videoul este ok pt a da load
    print('Error while loading the video!')
    sys.exit()

ok, frame = video.read()#o sa indice daca putem sa ii dam load la primul frame la video 
    #print(ok)
if not ok:
    print('Erro while loading the frame!')
    sys.exit()
#print(ok)
#bbox variabila care incepe pozitia obiectului in primul frame 
#gen cand dam run la code o sa inceapa cu primul frame al video-ului
#cand ne da return ne da coordonatele box ului + size-ul pe pixeli 
bbox = cv2.selectROI(frame) 
print(bbox)

ok = tracker.init(frame, bbox)#initializam alrg ->  punem pozitia
print(ok)

colors = (randint(0, 255), randint(0,255), randint(0, 255)) # RGB -> BGR
print(colors)
#pt ca pixelii lucreaza cu rbg, but opencv cu bgr 
while True:
    ok, frame = video.read() #mergem prin fiecare frame al video ului 
    #print(ok)
    if not ok: 
        break

    ok, bbox = tracker.update(frame)#asta pt ca ce am pus noi o sa se tot modifice 
    #ii dam update la fecare frame al video-ului
    #print(ok, bbox) 
    if ok == True:
        (x, y, w, h) = [int(v) for v in bbox]
         #punem fiecare value al box -ului in variabile
        #print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2)
       
    else:
        cv2.putText(frame, 'Tracking failure!', (100,80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255))

    cv2.putText(frame, tracker_type, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255))

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0XFF == 27: # esc key 
        break

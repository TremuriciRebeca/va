import numpy as np
import cv2

cap = cv2.VideoCapture('Videos/walking.avi')
#definim parametrii sub forma unui dictionar: ii dam 100 de colturi pe care sa le vada
#quality colturilor care vor fi detectate, cand ii dam run o sa caute 100 de colturi
#dar nu toate colturilor o sa die folosite deoarece ii dam un filtru pentru cele mai bune
#ii dam un scor cu quality 
#ex sa zicem ca cel mai bun colt are scorul 1500 -> multiplicam cu quality lvl = 450
#colturile care au mai < decat 450 vor fi sterse
#mindistance respecta distanta de pixeli pe care ii da de la un colt gasit ( 7 e recomandat)
parameters_shitomasi = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7)
parameters_lucas_kanade = dict(winSize = (15,15), maxLevel = 2,
#size ul minim o sa fie 15x15 225 pixeli ( recomandat ), dupa lvl ca in imagine l0, l1, l2.. 
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

colors = np.random.randint(0,255, (100, 3))
#print(np.shape(colors))
#print(colors)
#ii dam primul frame al videoului si trb sa il convertim in gray scailing ( e recomandat ca procesul sa fie mai rapid)
ok, frame = cap.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Init frame', frame_gray_init)
#cv2.waitKey(0)

#initializarea algoritmului, dandu-i parametrii primului frame al video-ului 
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask = None, **parameters_shitomasi)
print(len(edges))
print(edges)

mask = np.zeros_like(frame)
#print(np.shape(mask))
#print(mask)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None, **parameters_lucas_kanade)

    news = new_edges[status == 1]
    olds = edges[status == 1]

    for i, (new, old) in enumerate(zip(news, olds)):
       a, b = new.ravel()
       c, d = old.ravel()

       mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), colors[i].tolist(), 2)
       frame = cv2.circle(frame, (int(a),int(b)), 5, colors[i].tolist(), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('Optical flow sparse', img)
    if cv2.waitKey(1) == 13: # enter
        break

    frame_gray_init = frame_gray.copy()
    edges = news.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()

















import math
import cv2 as cv
import numpy as np
from numpy import random
 
capture = cv.VideoCapture('CarsDrivingUnderBridge.mp4')
#capture = cv.VideoCapture('circle_move.mp4')
 
 
backSub = cv.createBackgroundSubtractorMOG2(history=None,varThreshold=None,detectShadows=True)
 
print("===========SHOW DEFAULT PARAMETER===========") #Print default value
print(f"getHistory={backSub.getHistory()}\ngetNMixtures={backSub.getNMixtures()}\ngetDetectShadows={backSub.getDetectShadows()}\nvarThreshold={backSub.getVarThreshold()}")
#backSub.setHistory(600) # Amount of mean sigma of BG
#backSub.setNMixtures(2) # Number of Gaussian Distribution
backSub.setDetectShadows(True) # Detect Shadow
backSub.setVarThreshold(100) # Threshold Motion
 
class TrackedBlob:
    id = None
    color = None
    alive = True # if this flag false = marked to remove
    def __init__(self,blob_id,xywh):
        self.id = blob_id
        (self.x,self.y,self.w,self.h) = xywh
        c = random.randint(255, size=(3))
        self.color = (c[0].item(),c[1].item(),c[2].item())
        self.path = []
        self.nlost = 0
        self.allLife = 1
        self.matchedStatus = True 
        self.alreadyCounted = False # use for count function
    def XYWH(self):
        return [self.x,self.y,self.w,self.h]
    def calDiffBlobWithContour(self, contour_xywh, weightPosition = 0.5, weightScale = 0.5):
        (xContour,yContour,wContour,hContour) = contour_xywh
        return (math.sqrt((xContour - self.x)**2 + (yContour - self.y)**2) * weightPosition) + (math.sqrt((wContour - self.w)**2 + (hContour - self.h)**2) * weightScale)
    def updateBlob(self, status, xywh=[0,0,0,0]):
        if (status): # Matched
            self.nlost = 0
            self.allLife += 1
            self.path.append(xywh)
            (self.x,self.y,self.w,self.h) = xywh
        else : # Not Matched
            self.nlost += 1
            if(self.nlost >= 4): # if not matched xx frame will be marked to remove
                self.alive = False
    def setMatched(self):
        self.matchedStatus = True
    def resetMatched(self):
        self.matchedStatus = False
    def getMatched(self):
        return self.matchedStatus
    def getNLost(self):
        return self.nlost
    def setCounted(self): 
        self.alreadyCounted = True
    def getCounted(self): 
        return self.alreadyCounted
 
class BlobTracker:
    def __init__(self, distanceThreshold = 60):
        self.TrackedBlob_Table = []
        self.distanceThreshold = distanceThreshold
        self.lastID = 0
    def setDistanceThreshold(self, distanceThreshold = 60):
        self.distanceThreshold = distanceThreshold
    def trackXYWHs(self, ContourXYWHs): # ContourXYWHs -> Incomming Objects Table
        # if no TrackedBlob in Table
        if(len(self.TrackedBlob_Table)==0):
            for XYWH in ContourXYWHs:
                self.addNewTrackedBlob(XYWH)
        # Matching Incomming Objects with all
        else :
            for XYWH in ContourXYWHs: # accessing in each contour to update or new tracked blob
                minDistance = 1000000000 #         -1 = not match with anyone
                minDistanceTrackedBlobID = -1 #    -1 = not match with anyone
                for TrackedBlob in self.TrackedBlob_Table:
                    if(not TrackedBlob.getMatched()): # if didnot match yet
                       diffValue = TrackedBlob.calDiffBlobWithContour(XYWH)
                       #print(f'diffValue {diffValue} of Blob={XYWH} with TrackedBlob#{TrackedBlob.id}={TrackedBlob.XYWH()}({TrackedBlob.color})')
                       if(diffValue <= self.distanceThreshold and diffValue < minDistance): # ถ้าเข้าเกณฑ์ และมีค่าน้อยสุดก็ marked ไว้ก่อน
                           minDistanceTrackedBlobID = TrackedBlob.id
                           minDistance = diffValue
                # if matched some TrackedBlob -> update
                if(minDistanceTrackedBlobID!=-1):
                    self.updateTrackedBlob(minDistanceTrackedBlobID, XYWH)
                else:
                    self.addNewTrackedBlob(XYWH)
        # if some TrackedBlob has not be updated -> set Lost
        for TrackedBlob in self.TrackedBlob_Table:
            if(not TrackedBlob.getMatched()): # if didnot match yet
                TrackedBlob.updateBlob(False)
        # remove all dead (not alive)
        numTrackedBlob = len(self.TrackedBlob_Table)
        i=0
        while i < numTrackedBlob:
            if(not self.TrackedBlob_Table[i].alive): # if didnot match yet
                self.TrackedBlob_Table.pop(i) # remove
                numTrackedBlob-=1 # move down counter
                i-=1 # move down iterator
            i+=1
        # reset all Match Status
        for TrackedBlob in self.TrackedBlob_Table:
            TrackedBlob.resetMatched()
    def addNewTrackedBlob(self, XYWH):
        self.TrackedBlob_Table.append(TrackedBlob(self.lastID,XYWH))
        self.lastID += 1
    def updateTrackedBlob(self, id, XYWH):
        for TrackedBlob in self.TrackedBlob_Table:
            if TrackedBlob.id == id:
                TrackedBlob.updateBlob(True,XYWH) # update
                TrackedBlob.setMatched() # marked already Mateched
                break
    def drawTrackedBlob(self, image, fontSize=1.0, thickness=2):
        for TrackedBlob in self.TrackedBlob_Table:
            if TrackedBlob.getNLost()==0 and TrackedBlob.alive : # and TrackedBlob.allLife>1'''
                color = TrackedBlob.color
                (x,y,w,h) = TrackedBlob.XYWH()
                cv.rectangle(image, (x,y), (x+w,y+h), color, thickness)
                cv.putText(image, str(TrackedBlob.id), (x+2,y+2), cv.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness)
 
 
class BlobExtractor:
    def __init__(self):
        self.contours = None
        self.hierarchy = None
        self.XYWHs = None
        self.colors = None
    def execute(self,segmented_bin_img):
        edge16S_img = cv.Laplacian(segmented_bin_img, cv.CV_16S, ksize=3)
        edge_img = cv.convertScaleAbs(edge16S_img)
        self.contours, self.hierarchy = cv.findContours(edge_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.XYWHs = [ cv.boundingRect(contour) for contour in self.contours]
    def filterMinArea(self,min):
        ''' filter out(remove) countours that  have area < min'''
        temp_contours = []
        for i,cnt in enumerate(self.contours):
            (x,y,w,h) = cv.boundingRect(cnt)
            if((w*h)>=min):
                temp_contours.append(cnt) # [x,y,w,h]
        self.contours = temp_contours.copy()
        self.XYWHs = [ cv.boundingRect(contour) for contour in self.contours]
    def filterInArea(self,XYWH):
        ''' filter only countour in Area XYWH'''
        temp_contours = []
        (xmin,ymin) = XYWH[:2]
        xmax = xmin + XYWH[2]
        ymax = ymin + XYWH[3]
        for i,cnt in enumerate(self.contours):
            (x,y,w,h) = cv.boundingRect(cnt)
            if( (x>=xmin and x<=xmax) and (y>=ymin and y<=ymax)):
                temp_contours.append(cnt) # [x,y,w,h]
        self.contours = temp_contours.copy()
        self.XYWHs = [ cv.boundingRect(contour) for contour in self.contours]
    def getContours(self):
        return self.contours
    def getXYWHs(self):
        return self.XYWHs
 
distanceThreshold = 60  
mainBlobTracker = BlobTracker(distanceThreshold); # create main tracker
cv.namedWindow('Frame',cv.WINDOW_NORMAL)
def changeDistanceThreshold(x):
    global distanceThreshold
    distanceThreshold = cv.getTrackbarPos('DistanceThreshold','Frame')
    mainBlobTracker.setDistanceThreshold(distanceThreshold)
cv.createTrackbar('DistanceThreshold', 'Frame', distanceThreshold, 400, changeDistanceThreshold)
 
 y_startCount = 200
y_endCount = 400
carCount = 0 # number of cars which passed line 
 
while True:
    ret, frame = capture.read()
    id_frame = capture.get(cv.CAP_PROP_POS_FRAMES)
    if frame is None:
        break
 
    fgMask = backSub.apply(frame,learningRate=0.005) # learningRate
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(id_frame), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    _,fgMask = cv.threshold(fgMask, 200, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
    fgMask = cv.dilate(fgMask, kernel, iterations=1)
 
    Blob = BlobExtractor()
    Blob.execute(fgMask)
    Blob.filterMinArea(2000)
    FocusArea = [300,100,750,550]
    Blob.filterInArea(FocusArea)
    
 
    (W,H) = fgMask.shape
    contours_img = np.zeros((W,H,3),dtype=np.uint8)
    if(len(Blob.getContours())>=1):
        cv.drawContours(contours_img, Blob.getContours(), -1, (255,0,0)) 
 
    (xFA,yFA,wFA,hFA) = FocusArea
    cv.rectangle(frame, (xFA,yFA), (xFA+wFA,yFA+hFA), (255,255,255))
    cv.rectangle(contours_img, (xFA,yFA), (xFA+wFA,yFA+hFA), (255,255,255))
 
    if(id_frame>100):
        mainBlobTracker.trackXYWHs(Blob.getXYWHs())
        mainBlobTracker.drawTrackedBlob(frame,fontSize=1)
        mainBlobTracker.drawTrackedBlob(contours_img,fontSize=1)
 
    cv.imshow('Frame', frame)
    cv.imshow('FG_Mask', fgMask)
    cv.imshow('Contours', contours_img)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
 
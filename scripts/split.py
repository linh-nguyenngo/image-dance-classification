# Need to install opencv to use cv2: 'pip3 install opencv-python'
import cv2
import glob

# videos_path = './videos/zapin/'
# images_path = "./images/zapin/image%s.jpg"

# apsara_videos_path = './videos/apsara/'
# apsara_images_path = "./images/apsara/image%s.jpg"

# tarian_jorget_videos_path = './videos/tarian_jorget/'
# tarian_jorget_images_path = "./images/tarian_jorget/image%s.jpg"

khen_videos_path = './videos/predictions/apsara/'
khen_images_path = "./images/apsara_predict_1/image%s.jpg"

videos = glob.glob(khen_videos_path + '*.mp4')
frameRate = 1 #//it will capture image in each 0.5 second
count=1

def getFrame(vidcap, sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(khen_images_path %(str(count)), image) # save frame as JPG file
    return hasFrames

for video in videos:
    print('Processing %s' %(video))
    sec = 0
    vidcap = cv2.VideoCapture(video)
    success = getFrame(vidcap, sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(vidcap, sec)

print('Total %s images created.' %(count-1))
print('Done.')

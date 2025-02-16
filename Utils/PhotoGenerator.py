import cv2
import os
import subprocess
import platform

cam = cv2.VideoCapture("E:\\Dataset\\dataset_1\\keyframe_1\\data\\rgb.mp4")

try:
    if not os.path.exists("./data"):
        os.makedirs("./data")

except OSError:
    print("Error: Creating directory. " + 'data')

currentframe = 0

while(True):
    ret, frame = cam.read()
    if ret:
        height, width, _ = frame.shape

        top_half = frame[:height // 2, :]
        bottom_half = frame[height // 2:, :]

        left_name = './data/left_frame' + str(currentframe) + '.jpg'
        print('Creating...' + left_name)
        cv2.imwrite(left_name, top_half)

        right_name = './data/right_frame' + str(currentframe) + '.jpg'
        print('Creating...' + right_name)
        cv2.imwrite(right_name, bottom_half)

        currentframe += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()

if platform.system() == 'Windows':
    subprocess.Popen(['explorer', './data'])
elif platform.system() == 'Darwin':  # macOS
    subprocess.Popen(['open', './data'])
else:  # Linux
    subprocess.Popen(['xdg-open', './data'])
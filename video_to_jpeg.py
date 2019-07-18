import cv2
vidcap = cv2.VideoCapture('data/zTn-nNj5Bng_8_19.avi')
success, image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success, image = vidcap.read()
  print('Found a new frame: ', success)
  count += 1

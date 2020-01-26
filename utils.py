import datetime
from threading import Thread

import cv2
import time

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
        # self._end = datetime.datetime.now()

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
    
    def stop(self):
        # indicate that the thread should be stopped
        self._end = datetime.datetime.now()


 
class VideoStream:
    def __init__(self, src=0, fps=False):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            return
        if fps:
            self.fps = FPS()
        else:
            self.fps = None

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        if self.fps:
            self.fps.start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.grabbed:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            if self.fps:
                self.fps.update()
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.stream.isOpened(), self.frame

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()
        if self.fps:
            self.fps.stop()
            print("Approximate fps: {}".format(self.fps.fps()))
'''
Created on March 12, 2017

Converts the log.txt file from the Convolution Project to a csv file.
Sorts the frames according to the kernel type (so all frames with
kernel x are grouped together in csv).

@precondition: update path to directory of log.txt, file exists

@author: William Schneble
'''

import pandas as pd

#frame object
class frame():
    def __init__(self, frame, fps, mps, ktype, ktime):
        self.frame = cleanStr(frame)
        self.fps = cleanStr(fps)
        self.mps = cleanStr(mps)
        self.ktype = cleanStr(ktype)
        self.ktime = cleanStr(ktime)
    def __str__(self):
        return self.frame + ' ' + self.fps + ' ' + self.mps + ' ' + self.ktype + ' ' + self.ktime

#for sorting by kernel type
def getKey(frame):
    return frame.ktype

#get rid of the excess stuff
def cleanStr(str):
    tokens = str.split("]")
    return tokens[0]

#put the log data into frame objects
path = "C:/Users/Alex/Documents/Visual Studio 2015/Projects/convolutions/convolutions/"
listFrames = []
with open(path + 'log.txt') as f:
    for line in f:
        tokens = line.split(":")
        myframe = frame(tokens[1], tokens[2], tokens[3], tokens[4], tokens[5])
        listFrames.append(myframe)

#put the frame objects into a csv with similar frames together
stringList = []
listFrames = sorted(listFrames, key=getKey)
for i in listFrames:
    stringList.append([i.frame, i.fps, i.mps, i.ktype, i.ktime])
df = pd.DataFrame(stringList, columns=['Frame', 'FPS', 'MPS', 'Kernel Type', 'Kernel Time(ms)'])
df.set_index('Frame', inplace=True)
df.to_csv(path + 'results.csv')
import json


def extract_json(txt):
    left = 0
    right = 0
    for i, s in enumerate(txt):
        if s == '{':
            left += 1
            if left == 1:
                leftest = i
        if s == '}':
            right += 1
            if left > 0 and right == left:
                rightest = i
                return json.loads(txt[leftest:rightest + 1])
            
    print('Found no dict: {}-{}'.format(left, right))
    return {}


def unit_convert(B): # B (int)
    K = B / 1024
    if K >= 1024:
        M = K / 1024
        if M >= 1024:
            G = M / 1024
            return '%.2fGB' % G
        else:
            return '%.2fMB' % M
    else:
        return '%.2fKB' % K


class Reader():
    def __init__(self, content):
        self.content = content
        self.start = 0
        self.eof = False
        self.length = len(self.content)
        
    def read(self, n=1):
        if self.length > (self.start + n):
            out = self.content[self.start:self.start + n]
            self.start += n
        else:
            out = self.content[self.start:]
            self.eof = True
        return out
    

def add_flv(flv, target, videoTimeStamp, audioTimeStamp):
    with open(flv, 'rb') as f:
        content = f.read()
    reader = Reader(content)
    
    header = reader.read(13)

    with open(target, 'ab') as f:
        while not reader.eof:
            dataType = reader.read(1)
            dataSize = reader.read(3)
            timeStamp = int.from_bytes(reader.read(3), 'big')
            headerRemained = reader.read(4)

            if dataType == b'\t': # video
                timeStamp += videoTimeStamp
                videoTS = timeStamp
            if dataType == b'\x08': # audio
                timeStamp += audioTimeStamp
                audioTS = timeStamp
            timeStamp = timeStamp.to_bytes(3, 'big')

            tagHeader = dataType + dataSize + timeStamp + headerRemained
            tagData_andSize = reader.read(int.from_bytes(dataSize, 'big') + 4)

            f.write(tagHeader)
            f.write(tagData_andSize)
        
    return videoTS, audioTS
        
    
def merge_flv(flvs, target):
    flvs = sorted(flvs)
    videoTS = 0
    audioTS = 0
    for i, flv in enumerate(flvs):
        with open(flv, 'rb') as f:
            content = f.read()
        reader = Reader(content)
        
        header = reader.read(13) # flvHeader + tagSize0
        if i == 0:
            with open(target, 'wb') as f:
                f.write(header)
                
        videoTS, audioTS = add_flv(flv, target, videoTS, audioTS)
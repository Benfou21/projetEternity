import numpy as np

class piece:
    def __init__(self,id,colors,rot):
        self.id = id
        self.colors = colors
        self.rot = rot


arr =  []

for i in range(3):
    arr.append([])
    for j in range(3):
        arr[i].append( 0)



arr[0][0] =   piece(1,[0,1,2,3],1)


print([i for i in range(4,9)])
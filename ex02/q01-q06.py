import numpy as np
#שאלה 1
blackImg=np.zeros((100,200),dtype=np.uint8)

#שאלה 2
whiteImg=np.full((100,200),255,dtype=np.uint8)

#שאלה 3
blueImg=np.full((100,200),128,dtype=np.uint8)

#שאלה 4
print(blackImg.shape) #(100,200)
print(blackImg.ndim) #2
print(blackImg.size) #20000
print(blackImg.nbytes) #20000

print(whiteImg.shape) #(100,200)
print(whiteImg.ndim) #2
print(whiteImg.size) #20000
print(whiteImg.nbytes) #20000

print(blueImg.shape) #(100,200)
print(blueImg.ndim) #2
print(blueImg.size) #20000
print(blueImg.nbytes) #20000

#שאלה 5
print(np.zeros((3000, 4000, 3), dtype=np.uint8).nbytes)  # 36000000 36mb
print(np.zeros((1080, 1920, 3), dtype=np.uint8).nbytes)  # 6220800 6.22mb

#שאלה 6
b = np.arange(12).reshape(3, 4)
print(b) #[[ 0  1  2  3]
 #[ 4  5  6  7]
# [ 8  9 10 11]]
print(b.sum()) #66
print(b.sum(axis=0)) #[12 15 18 21]
print(b.sum(axis=1)) #[ 6 22 38]
print(b[2, 3], b[-1, -1]) #11 11


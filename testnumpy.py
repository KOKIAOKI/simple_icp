import numpy as np

class Array2D:
    def __init__(self):
        self.x = np.empty((0,1))
        self.y = np.empty((0,1))
        self.ev = np.empty((0,1))



if __name__ == "__main__":
    offset_array = Array2D()
    print(offset_array.x)

    offset_array.x = np.append(offset_array.x, np.array([[1]]), axis = 0)
    print(offset_array.x)

    offset_array.x = np.append(offset_array.x, np.array([[2]]), axis = 0)
    print(offset_array.x)
import numpy as np

class CircularBuffer:
    def __init__(self, arr):
        self.max_size = arr.shape[0]
        self.__arr = arr
        self.i = 0
        self.cur_size = 0

    def add(self, item):
        self.__arr[self.i] = item
        self.i += 1
        
        if self.i >= self.max_size:
            self.i = 0

        if self.cur_size < self.max_size:
            self.cur_size += 1

    def get_arr(self):
        if self.cur_size == self.max_size:
            return self.__arr

        return self.__arr[:self.cur_size]

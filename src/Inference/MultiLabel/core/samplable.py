# -*- coding: utf-8 -*-

class Samplable:
    """
    The base class for the object that will use sampling method
    """
    def __init__(self ):
        self.current = -1
        self.objs = []

    def append(self, obj):
        self.objs.append(obj)
        self.current += 1

    def setV(self, val, index = 0):
        if (index == 0):
            index = self.current
        self.objs[index] = val

    def getV(self, index = 0):
        if (index == 0):
            index = self.current
        return self.objs[index]

    def print_obj(self, index = 0):
        if (index == 0):
            index = self.current
        print(self.objs[index], end=' ')


class RealV(Samplable):

    def __init__(self, val):
        Samplable.__init__(self)
        self.append(val)

    def getV(self, index = 0):
        if (index == 0):
            index = self.current
        return float(self.objs[index])


class IntV(Samplable):
    def __init__(self, val):
        Samplable.__init__(self)
        self.append(val)

    def getV(self, index=0):
        if (index == 0):
            index = self.current
        return int(self.objs[index])

    def get_max_type(self, begin):
        internal_map = dict()
        list_len = len(self.objs)
        max_val = 0
        max_item = 0
        for i in range(begin, list_len):
            if self.objs[i] not in internal_map:
                internal_map.setdefault(self.objs[i], 1)
            else:
                internal_map[self.objs[i]] += 1
        for (k, v) in internal_map.items():
            if v > max_val:
                max_val = v
                max_item = k
        return max_item


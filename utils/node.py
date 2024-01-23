class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]

        if len(n) == 3:
            self.yaw = n[2]
        else:
            self.yaw = None

        self.parent = None
        self.cost = 0
        #self.StateTraj = None
        self.childrenNodeInds = set([])
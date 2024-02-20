
class Env:
    def __init__(self):
        WIDTH = 15
        HEIGHT = 15
        self.x_range = (0, WIDTH)
        self.y_range = (0, HEIGHT)
        self.obs_boundary = self.obs_boundary(WIDTH, HEIGHT)
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary(width, height):  # circle
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        w = width
        h = height
        obs_boundary = [
            [0, 0, 1, h],
            [0, h, w, 1],
            [1, 0, w, 1],
            [w, 1, 1, h]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        # obs_rectangle = [
        #     [14, 12, 8, 2],
        #     [18, 22, 8, 3],
        #     [26, 7, 2, 12],
        #     [32, 14, 10, 2]
        # ]
        obs_rectangle = []
        return obs_rectangle

    # @staticmethod
    # def obs_circle():
    #     obs_cir = [
    #         [7, 12, 3],
    #         [46, 10, 2],
    #         [25, 10, 3],
    #         [15, 5, 2],
    #         [15, 15, 2],
    #         [37, 7, 3],
    #         [37, 23, 3]
    #     ]

    #     return obs_cir
    
    @staticmethod
    def obs_circle():
        obs_cir = [
            [10, 10, 3],
            [10, 15, 3],
            [10, 20, 3],
            [10, 25, 3],
            [20, 5, 3],
            [20, 10, 3],
            [20, 15, 3],
            [20, 20, 3],
            [30, 5, 3],
            [30, 10, 3],
            [35, 15, 3],
            [40, 20, 3],
            [45, 25, 3]
        ]
        obs_cir = [
            [7.5, 2, 1],
            [7.5, 4, 1],
            [7.5, 6, 1],
            [7.5, 8, 1],
            [12.0, 10.0, 1],

        ]
        # randomly generate 6 obstacles
        # import random
        # obs_cir = []
        # for i in range(6):
        #     x = random.uniform(0, 10)
        #     y = random.uniform(0, 10)
        #     r = random.uniform(1, 1)
        #     obs_cir.append([x, y, r])

       #obs_cir = [[100, 100, 1]]

        return obs_cir
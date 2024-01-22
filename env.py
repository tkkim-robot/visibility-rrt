class Env:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():  # circle
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
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
        obs_cir = [[100, 100, 1]]

        return obs_cir
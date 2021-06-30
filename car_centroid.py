import numpy as np


class CarCentroid:
    """
        Class to track cars in a video
    """
    __next_id = 1
    __buffer_size = 20

    def __init__(self, x, y, fps):
        self.x = x
        self.y = y
        self.id = CarCentroid.__next_id
        self.fps = fps
        self.direction = ''
        CarCentroid.__next_id += 1

        self.prev_positions = [np.array([x, y])]

    def update_position(self, new_position):
        """
        Update the current position of the car
        :param new_position:
        :return:
        """
        x, y = new_position
        self.x = x
        self.y = y

        # check if the buffer is full
        if len(self.prev_positions) == CarCentroid.__buffer_size:
            self.prev_positions.pop(0)

        self.prev_positions.append(np.array(new_position))
        self.__update_direction()

    def __update_direction(self):
        """
        Update movements direction of the car
        """
        mnt_thres = 5

        if len(self.prev_positions) < 2:
            self.direction = ''
            return

        diff_x = self.prev_positions[len(self.prev_positions) - 1][0] - self.prev_positions[0][0]
        diff_y = self.prev_positions[len(self.prev_positions) - 1][1] - self.prev_positions[0][1]

        dir_x, dir_y = '', ''
        # only update the direction if the movement was significant
        if np.abs(diff_x) > mnt_thres:
            dir_x = 'Right' if np.sign(diff_x) == 1 else 'Left'
        if np.abs(diff_y) > mnt_thres:
            dir_y = 'Down' if np.sign(diff_y) == 1 else 'Up'

        # handle movement in both directions
        if dir_x != "" and dir_y != "":
            self.direction = f"{dir_y}-{dir_x}"
        else:
            self.direction = dir_x if dir_x != "" else dir_y

    def get_avg_speed(self):
        """
        Returns average speed for last __buffer_size positions
        :return: average speed pixels/sec
        """
        speed_thres = 15

        if len(self.prev_positions) < 2:
            return 0

        # compute speed between past frames
        speed_arr = []
        for i in range(len(self.prev_positions) - 1):
            speed_arr.append(np.linalg.norm(self.prev_positions[i] - self.prev_positions[i - 1]))

        # take the mean between frames' speed
        avg_speed = int(np.mean(speed_arr) * self.fps)

        # only output if speed is significant
        return avg_speed if avg_speed > speed_thres else 0

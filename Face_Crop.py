import numpy
import cv2

class face_crop_level1():

    def __init__(self, data):
        self.num = data.shape[0]
        self.height = self.width = int(numpy.sqrt(data.shape[1]))
        self.image = numpy.reshape(data, (-1, self.height, self.width, 1))

    def eye_nose_crop(self):
        eye_nose_start = int(self.height * 0.1)
        eye_nose_end = int(self.height * 0.7)
        eye_nose_height = eye_nose_end - eye_nose_start
        eye_nose_data = numpy.array((self.num, eye_nose_height, self.width, 1))
        eye_nose_data = self.image[:, eye_nose_start: eye_nose_end, :, :]
        eye_nose_top_left = eye_nose_start
        return eye_nose_data, eye_nose_top_left


    def nose_mouth_crop(self):
        nose_mouth_start = int(self.height * 0.5)
        nose_mouth_end = int(self.height * 1.0)
        nose_mouth_height = nose_mouth_end - nose_mouth_start
        nose_mouth_data = numpy.array((self.num, nose_mouth_height, self.width, 1))
        nose_mouth_data = self.image[:, nose_mouth_start: nose_mouth_end, :, :]
        nose_mouth_top_left = nose_mouth_start
        return nose_mouth_data, nose_mouth_top_left

class face_crop_level2():

    def __init__(self, image_data, coor_data):
        self.height = self.width = numpy.sqrt(image_data.shape[1])
        self.num = image_data.shape[0]
        self.image = numpy.array(image_data, (-1, self.height, self.width, 1))
        self.coor = coor_data

    def left_eye_crop(self):
        left_eye_data = numpy.array((self.num, 20, 20, 1))
        left_eye_x = self.coor[:, 0]
        left_eye_y = self.coor[:, 1]
        left_eye_data = self.image[:, left_eye_x - 10: left_eye_x + 10, left_eye_y - 10: left_eye_y + 10, :]
        left_eye_top_left = numpy.array((self.num, 2))
        left_eye_top_left[:, 0], left_eye_top_left[:, 1] = left_eye_x - 10, left_eye_y - 10
        return left_eye_data, left_eye_top_left

    def right_eye_crop(self):
        right_eye_data = numpy.array((self.num, 20, 20, 1))
        right_eye_x = self.coor[:, 2]
        right_eye_y = self.coor[:, 3]
        right_eye_data = self.image[:, right_eye_x - 10: right_eye_x + 10, right_eye_y - 10: right_eye_y + 10, :]
        right_eye_top_left = numpy.array((self.num, 2))
        right_eye_top_left[:, 0], right_eye_top_left[:, 1] = right_eye_x - 10, right_eye_y - 10
        return right_eye_data, right_eye_top_left

    def nose_crop(self):
        nose_data = numpy.array((self.num, 20, 30, 1))
        nose_x = self.coor[:, 4]
        nose_y = self.coor[:, 5]
        nose_data = self.image[:, nose_x - 10: nose_x + 10, nose_y - 15: nose_y + 15, :]
        nose_top_left = numpy.array((self.num, 2))
        nose_top_left[:, 0], nose_top_left[:, 1] = nose_x - 10, nose_y - 10
        return nose_data, nose_top_left

    def left_mouth_crop(self):
        left_mouth_data = numpy.array((self.num, 20, 20, 1))
        left_mouth_x = self.coor[:, 6]
        left_mouth_y = self.coor[:, 7]
        left_mouth_data = self.image[:, left_mouth_x - 10: left_mouth_x + 10, left_mouth_y - 10: left_mouth_y + 10, :]
        left_mouth_top_left = numpy.array((self.num, 2))
        left_mouth_top_left[:, 0], left_mouth_top_left[:, 1] = left_mouth_x - 10, left_mouth_y - 10
        return left_mouth_data, left_mouth_top_left

    def right_mouth_crop(self):
        right_mouth_data = numpy.array((self.num, 20, 20, 1))
        right_mouth_x = self.coor[:, 8]
        right_mouth_y = self.coor[:, 9]
        right_mouth_data = self.image[:, right_mouth_x - 10: right_mouth_x + 10, right_mouth_y - 10: right_mouth_y + 10, :]
        right_mouth_top_left = numpy.array((self.num, 2))
        right_mouth_top_left[:, 0], right_mouth_top_left[:, 1] = right_mouth_x - 10, right_mouth_y - 10
        return right_mouth_data, right_mouth_top_left

class face_crop_level3():

    def __init__(self, image_data, coor_data):
        self.height = self.width = int(numpy.sqrt(image_data.shape[1]))
        self.num = image_data.shape[0]
        self.image = numpy.array(image_data, (-1, self.height, self.width, 1))
        self.coor = coor_data

    def left_eye_crop(self):
        left_eye_data = numpy.array((self.num, 10, 10, 1))
        left_eye_x = self.coor[:, 0]
        left_eye_y = self.coor[:, 1]
        left_eye_data = self.image[:, left_eye_x - 5: left_eye_x + 5, left_eye_y - 5: left_eye_y + 5, :]
        left_eye_top_left = numpy.array((self.num, 2))
        left_eye_top_left[:, 0], left_eye_top_left[:, 1] = left_eye_x - 10, left_eye_y - 10
        return left_eye_data, left_eye_top_left

    def right_eye_crop(self):
        right_eye_data = numpy.array((self.num, 10, 10, 1))
        right_eye_x = self.coor[:, 2]
        right_eye_y = self.coor[:, 3]
        right_eye_data = self.image[:, right_eye_x - 5: right_eye_x + 5, right_eye_y - 5: right_eye_y + 5, :]
        right_eye_top_left = numpy.array((self.num, 2))
        right_eye_top_left[:, 0], right_eye_top_left[:, 1] = right_eye_x - 10, right_eye_y - 10
        return right_eye_data, right_eye_top_left

    def nose_crop(self):
        nose_data = numpy.array((self.num, 20, 30, 1))
        nose_x = self.coor[:, 4]
        nose_y = self.coor[:, 5]
        nose_data = self.image[:, nose_x - 5: nose_x + 5, nose_y - 10: nose_y + 10, :]
        nose_top_left = numpy.array((self.num, 2))
        nose_top_left[:, 0], nose_top_left[:, 1] = nose_x - 10, nose_y - 10
        return nose_data, nose_top_left

    def left_mouth_crop(self):
        left_mouth_data = numpy.array((self.num, 20, 20, 1))
        left_mouth_x = self.coor[:, 6]
        left_mouth_y = self.coor[:, 7]
        left_mouth_data = self.image[:, left_mouth_x - 5: left_mouth_x + 5, left_mouth_y - 5: left_mouth_y + 5, :]
        left_mouth_top_left = numpy.array((self.num, 2))
        left_mouth_top_left[:, 0], left_mouth_top_left[:, 1] = left_mouth_x - 10, left_mouth_y - 10
        return left_mouth_data, left_mouth_top_left

    def right_mouth_crop(self):
        right_mouth_data = numpy.array((self.num, 20, 20, 1))
        right_mouth_x = self.coor[:, 8]
        right_mouth_y = self.coor[:, 9]
        right_mouth_data = self.image[:, right_mouth_x - 5: right_mouth_x + 5, right_mouth_y - 5: right_mouth_y + 5, :]
        right_mouth_top_left = numpy.array((self.num, 2))
        right_mouth_top_left[:, 0], right_mouth_top_left[:, 1] = right_mouth_x - 10, right_mouth_y - 10
        return right_mouth_data, right_mouth_top_left


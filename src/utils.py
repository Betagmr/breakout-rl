import cv2

from src.const import PLATFORM_LENGTH


def get_platform_cords(map_area):
    down_area = map_area[150:151, :]

    for index, pixel in enumerate(down_area.reshape(-1)):
        if pixel != 0:
            left_side = index
            rigth_side = index + PLATFORM_LENGTH
            center = left_side + PLATFORM_LENGTH // 2

            return left_side, rigth_side, center 

    raise Exception("Platform not found")


def get_map(state_img):
    gray = cv2.cvtColor(state_img, cv2.COLOR_BGR2GRAY)
    return gray[40:195, 8:152]


def get_ball_position(current_state, previous_state):
    state = cv2.subtract(current_state, previous_state)

    for y_index, row in enumerate(state):
        for x_index, pixel in enumerate(row):
            if pixel != 0:
                return x_index, y_index, 

    return None

import cv2
import numpy as np
import pyautogui
import keyboard
import win32api
import win32con
import time
from mss import mss

# Constants
TARGET_COLOR = (243, 72, 72)  # Red
MIN_WIDTH, MAX_WIDTH = 97, 127
MIN_HEIGHT, MAX_HEIGHT = 11, 45
MIN_TARGET_PIXELS = 11
MIN_AREA = 1125
TOLERANCE = 5
MAX_DISTANCE = 100

def is_within_allowed_area(x, y, w, h, screen_width, screen_height):
    return not (
        (x < 400 and y < 400) or
        (x < 875 and y > screen_height - 300) or
        (x > screen_width - 725 and y > screen_height - 150) or
        (x > screen_width - 500 and y < 300)
    )

def lerp(start, end, t):
    return start + t * (end - start)

def smooth_move(prev_pos, target_pos, factor=0.4):
    if prev_pos is None:
        return target_pos
    return (
        int(prev_pos[0] + (target_pos[0] - prev_pos[0]) * factor),
        int(prev_pos[1] + (target_pos[1] - prev_pos[1]) * factor)
    )

def find_unit_pos(image, black_mask, target_color_mask):
    screen_height, screen_width = image.shape[:2]
    mouse_x, mouse_y = pyautogui.position()
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    unit_positions = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(int)
        x, y, w, h = cv2.boundingRect(box)

        if (MIN_WIDTH <= w <= MAX_WIDTH and 
            MIN_HEIGHT <= h <= MAX_HEIGHT and 
            w * h >= MIN_AREA and 
            is_within_allowed_area(x, y, w, h, screen_width, screen_height)):

            roi_target = target_color_mask[y:y+h, x:x+w]
            target_pixels = np.sum(roi_target) // 255

            if target_pixels >= MIN_TARGET_PIXELS and np.mean(black_mask[y:y+h, x:x+w]) > 20:
                center_x = x + w // 2
                t = (center_x - 0) / screen_width
                offset = lerp(75, -75, t)
                center_x += int(offset)

                center_y = y + h // 2 + 79
                distance = ((center_x - mouse_x)**2 + (center_y - mouse_y)**2)**0.5
                unit_positions.append((x, y, w, h, center_x, center_y, distance))

    unit_positions.sort(key=lambda pos: pos[6])
    return unit_positions

def find_units_near_previous(unit_positions, previous_location):
    if not previous_location:
        return []

    prev_x, prev_y = previous_location
    return sorted([
        (unit[0], unit[1], unit[2], unit[3], unit[4], unit[5], 
         ((unit[4] - prev_x)**2 + (unit[5] - prev_y)**2)**0.5) 
        for unit in unit_positions if ((unit[4] - prev_x)**2 + (unit[5] - prev_y)**2)**0.5 < MAX_DISTANCE
    ], key=lambda pos: pos[6])

def create_color_mask(image, color, tolerance=TOLERANCE):
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = color_hsv

    h_low = (h - tolerance) % 180
    h_high = (h + tolerance) % 180
    s_low, s_high = max(0, s - tolerance), min(255, s + tolerance)
    v_low, v_high = max(0, v - tolerance), min(255, v + tolerance)

    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

    if h_low < h_high:
        mask = cv2.inRange(image, lower, upper)
    else:
        mask1 = cv2.inRange(image, np.array([0, s_low, v_low], dtype=np.uint8), upper)
        mask2 = cv2.inRange(image, lower, np.array([179, s_high, v_high], dtype=np.uint8))
        mask = cv2.bitwise_or(mask1, mask2)

    return mask

def main():
    previous_unit_location = None

    with mss() as sct:
        monitor = sct.monitors[1]

        while True:
            if keyboard.is_pressed('del'):
                break

            if win32api.GetKeyState(win32con.VK_XBUTTON2) < 0:
                screenshot = np.array(sct.grab(monitor))
                image = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                target_mask = create_color_mask(image_hsv, TARGET_COLOR, TOLERANCE)
                black_mask = cv2.inRange(image_hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 30], dtype=np.uint8))

                unit_positions = find_unit_pos(image, black_mask, target_mask)

                if previous_unit_location:
                    near_units = find_units_near_previous(unit_positions, previous_unit_location)
                    if near_units:
                        closest_unit = near_units[0]
                        target_x, target_y = closest_unit[4], closest_unit[5]
                    else:
                        target_x, target_y = None, None
                else:
                    if unit_positions:
                        closest_unit = unit_positions[0]
                        target_x, target_y = closest_unit[4], closest_unit[5]
                    else:
                        target_x, target_y = None, None

                if target_x is not None and target_y is not None:
                    target_x, target_y = smooth_move(previous_unit_location, (target_x, target_y))
                    win32api.SetCursorPos((target_x, target_y))
                    previous_unit_location = (target_x, target_y)
                else:
                    previous_unit_location = None
            else:
                previous_unit_location = None

            time.sleep(0.01)

if __name__ == "__main__":
    main()

import pyautogui
import cv2
import numpy as np
import keyboard
import win32api
import win32con
import time
import mss

def is_within_allowed_area(x, y, w, h, screen_width, screen_height):
    # Personalize conforme necessário
    return not (
        (x < 400 and y < 400) or
        (x < 875 and y > screen_height - 300) or
        (x > screen_width - 725 and y > screen_height - 150) or
        (x > screen_width - 500 and y < 300)
    )

def lerp(start, end, t):
    return start + t * (end - start)

def find_unit_pos(image, black_mask, target_color_mask, min_width=0, max_width=750, min_height=0, max_height=750, min_area=375, min_target_pixels=2):
    screen_height, screen_width = image.shape[:2]
    mouse_x, mouse_y = pyautogui.position()

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    unit_positions = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        x, y, w, h = cv2.boundingRect(box)

        if min_width <= w <= max_width and min_height <= h <= max_height and w * h >= min_area:
            if is_within_allowed_area(x, y, w, h, screen_width, screen_height):
                roi_target = target_color_mask[y:y+h, x:x+w]
                target_pixels = np.sum(roi_target) // 255

                if target_pixels >= min_target_pixels and np.mean(black_mask[y:y+h, x:x+w]) > 20:
                    center_x = x + w // 2
                    t = (center_x - 0) / screen_width
                    offset = lerp(75, -75, t)
                    center_x += int(offset)

                    center_y = y + h // 2 + 79
                    distance = ((center_x - mouse_x)**2 + (center_y - mouse_y)**2)**0.5
                    unit_positions.append((x, y, w, h, center_x, center_y, distance))

    unit_positions.sort(key=lambda pos: pos[6])
    return unit_positions

def find_units_near_previous(unit_positions, previous_location, max_distance=100):
    if previous_location is None:
        return []

    prev_x, prev_y = previous_location
    near_units = []

    for unit in unit_positions:
        unit_x, unit_y = unit[4], unit[5]
        distance = ((unit_x - prev_x)**2 + (unit_y - prev_y)**2)**0.5
        if distance < max_distance:
            near_units.append((unit[0], unit[1], unit[2], unit[3], unit_x, unit_y, distance))

    near_units.sort(key=lambda pos: pos[6])
    return near_units

def create_color_mask(image, color, tolerance=10):
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = map(int, color_hsv)

    h_low = (h - tolerance) % 180
    h_high = (h + tolerance) % 180

    s_low = max(0, s - tolerance)
    s_high = min(255, s + tolerance)
    v_low = max(0, v - tolerance)
    v_high = min(255, v + tolerance)

    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

    if h_low < h_high:
        mask = cv2.inRange(image, lower, upper)
    else:
        mask1 = cv2.inRange(image, np.array([0, s_low, v_low], dtype=np.uint8),
                            np.array([h_high, s_high, v_high], dtype=np.uint8))
        mask2 = cv2.inRange(image, np.array([h_low, s_low, v_low], dtype=np.uint8),
                            np.array([179, s_high, v_high], dtype=np.uint8))
        mask = cv2.bitwise_or(mask1, mask2)

    return mask

def main():
    target_color = (243, 72, 72)  # Red
    min_width, max_width = 97, 127
    min_height, max_height = 11, 45
    min_target_pixels = 11
    min_area = 1125
    tolerance = 5

    previous_unit_location = None

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while True:
            if keyboard.is_pressed('del'):
                break

            if win32api.GetKeyState(win32con.VK_XBUTTON2) < 0:
                screenshot = np.array(sct.grab(monitor))
                image = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                target_mask = create_color_mask(image_hsv, target_color, tolerance)
                black_mask = cv2.inRange(image_hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 30], dtype=np.uint8))

                unit_positions = find_unit_pos(image, black_mask, target_mask, min_width, max_width, min_height, max_height, min_area, min_target_pixels)

                if previous_unit_location:
                    near_units = find_units_near_previous(unit_positions, previous_unit_location, max_distance=100)
                    if near_units:
                        closest_unit = near_units[0]
                        target_x, target_y = closest_unit[4], closest_unit[5]
                    else:
                        previous_unit_location = None

                if not previous_unit_location:
                    if unit_positions:
                        closest_unit = unit_positions[0]
                        target_x, target_y = closest_unit[4], closest_unit[5]
                    else:
                        target_x, target_y = None, None

                if target_x is not None and target_y is not None:
                    if previous_unit_location:
                        prev_x, prev_y = previous_unit_location
                        delta_x = target_x - prev_x
                        delta_y = target_y - prev_y
                        target_x += int(delta_x * 0.4)
                        target_y += int(delta_y * 0.4)

                    win32api.SetCursorPos((target_x, target_y))
                    previous_unit_location = (target_x, target_y)
                else:
                    previous_unit_location = None
            else:
                previous_unit_location = None

if __name__ == "__main__":
    main()

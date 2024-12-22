import pyautogui
import cv2
import numpy as np
import keyboard
import win32api
import win32con
import time
import mss


def is_within_allowed_area(x, y, w, h, screen_width, screen_height):
    return True


def lerp(start, end, t):
    return start + t * (end - start)


def create_color_mask(image_hsv, color, tolerance=10):
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = map(int, color_hsv)

    h_low, h_high = (h - tolerance) % 180, (h + tolerance) % 180
    s_low, s_high = max(0, s - tolerance), min(255, s + tolerance)
    v_low, v_high = max(0, v - tolerance), min(255, v + tolerance)

    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

    if h_low < h_high:
        return cv2.inRange(image_hsv, lower, upper)

    mask1 = cv2.inRange(image_hsv, np.array([0, s_low, v_low], dtype=np.uint8),
                        np.array([h_high, s_high, v_high], dtype=np.uint8))
    mask2 = cv2.inRange(image_hsv, np.array([h_low, s_low, v_low], dtype=np.uint8),
                        np.array([179, s_high, v_high], dtype=np.uint8))
    return cv2.bitwise_or(mask1, mask2)


def find_unit_pos(image, black_mask, target_color_mask, min_width, max_width, min_height, max_height, min_area, min_target_pixels):
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    screen_height, screen_width = image.shape[:2]
    mouse_x, mouse_y = pyautogui.position()

    unit_positions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

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
                    distance = ((center_x - mouse_x) ** 2 + (center_y - mouse_y) ** 2) ** 0.5
                    unit_positions.append((x, y, w, h, center_x, center_y, distance))

    unit_positions.sort(key=lambda pos: pos[6])
    return unit_positions


def find_nearest_units(unit_positions, previous_location, max_distance):
    if not previous_location:
        return []

    prev_x, prev_y = previous_location
    return sorted(
        [
            unit for unit in unit_positions
            if ((unit[4] - prev_x) ** 2 + (unit[5] - prev_y) ** 2) ** 0.5 < max_distance
        ],
        key=lambda pos: pos[6]
    )


def main():
    target_color = (243, 72, 72)
    min_width, max_width = 97, 127
    min_height, max_height = 11, 45
    min_area, min_target_pixels, tolerance = 1125, 11, 5
    max_distance = 75
    previous_unit_location = None

    print("Hold mouse5 to aim at the closest unit to mouse")
    print("Press 'del' to exit.")

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
                black_mask = cv2.inRange(image_hsv, np.array([0, 0, 0], dtype=np.uint8),
                                         np.array([180, 255, 30], dtype=np.uint8))

                unit_positions = find_unit_pos(
                    image, black_mask, target_mask, min_width, max_width,
                    min_height, max_height, min_area, min_target_pixels
                )

                if previous_unit_location:
                    near_units = find_nearest_units(unit_positions, previous_unit_location, max_distance)
                    if near_units:
                        target_x, target_y = near_units[0][4], near_units[0][5]
                    else:
                        previous_unit_location = None
                else:
                    if unit_positions:
                        target_x, target_y = unit_positions[0][4], unit_positions[0][5]
                    else:
                        target_x, target_y = None, None

                if target_x and target_y:
                    win32api.SetCursorPos((target_x, target_y))
                    previous_unit_location = (target_x, target_y)
            else:
                previous_unit_location = None
                time.sleep(0.025)


if __name__ == "__main__":
    main()

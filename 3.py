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
    # return not (
    #     (x < 400 and y < 400) or
    #     (x < 875 and y > screen_height - 300) or
    #     (x > screen_width - 725 and y > screen_height - 150) or
    #     (x > screen_width - 500 and y < 300)
    # )
 
def lerp(start, end, t):
    return start + t * (end - start)
 
def find_unit_pos(image, black_mask, target_color_mask, min_width=0, max_width=1000, min_height=0, max_height=1000, min_area=500, min_target_pixels=2):
    screen_height, screen_width = image.shape[:2]
    mouse_x, mouse_y = pyautogui.position()
 
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # Debugging
    # cv2.imshow("Black Mask", black_mask)
    # contour_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("Contours", contour_image)
    # cv2.waitKey(1)
 
    unit_positions = []
    for contour in contours:
        # Find the minimum area rectangle that fits the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        x, y, w, h = cv2.boundingRect(box)
 
        # Ensure the rectangle fits within width, height, and area requirements
        if min_width <= w <= max_width and min_height <= h <= max_height and w * h >= min_area:
            if is_within_allowed_area(x, y, w, h, screen_width, screen_height):
                roi_target = target_color_mask[y:y+h, x:x+w]
                target_pixels = np.sum(roi_target) // 255
 
                if target_pixels >= min_target_pixels and np.mean(black_mask[y:y+h, x:x+w]) > 20:  # Ensure there are at least min_target_pixels inside the black border and avoid transparent borders
                    center_x = x + w // 2
                    t = (center_x - 0) / screen_width  # Normalize center_x to a value between 0 and 1
                    offset = lerp(100, -100, t)  # Linearly interpolate between -100 and 100 based on the normalized position
                    center_x += int(offset)
 
                    center_y = y + h // 2 + 105  # Add 105 pixels below the health bar to find the unit position
                    distance = ((center_x - mouse_x)**2 + (center_y - mouse_y)**2)**0.5
                    unit_positions.append((x, y, w, h, center_x, center_y, distance))
 
    unit_positions.sort(key=lambda pos: pos[6])
    return unit_positions
 
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
 
def main():
    # For enemy in practice testing
    # target_color = (151, 81, 255) # Purple
    # min_width, max_width = 75, 150
    # min_height, max_height = 5, 20
 
    # For self in practice testing
    # target_color = (242,255,0) # Yellow
    # min_width, max_width = 130, 170
    # min_height, max_height = 15, 60
 
    # For enemy in real game 
    target_color = (243, 72, 72) # Red
    min_width, max_width = 130, 170
    min_height, max_height = 15, 60
 
    min_target_pixels = 15
    min_area = 1500
    tolerance = 5
    
    previous_unit_location = None
    
    print("Hold mouse5 to aim at the closest unit to mouse")
    print("Press 'del' to exit.")
 
    with mss.mss() as sct:
        monitor = sct.monitors[1]
 
        while True:
            if keyboard.is_pressed('del'):
                break
                    
            if win32api.GetKeyState(win32con.VK_XBUTTON2) < 0:
                loop_start_time = time.time()
                
                screenshot = np.array(sct.grab(monitor))
                # screenshot_elapsed = time.time() - loop_start_time
                # print(f"Screenshot lapsed time: {screenshot_elapsed:.4f}s")
 
                image = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # conversion_elapsed = time.time() - loop_start_time
                # print(f"Image conversion lapsed time: {conversion_elapsed:.4f}s")
 
                # Create masks once per frame
                target_mask = create_color_mask(image_hsv, target_color, tolerance)
                black_mask = cv2.inRange(image_hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 30], dtype=np.uint8))
                # mask_elapsed = time.time() - loop_start_time
                # print(f"Mask creation lapsed time: {mask_elapsed:.4f}s")
 
                # Find unit positions
                unit_positions = find_unit_pos(image, black_mask, target_mask, min_width, max_width, min_height, max_height, min_area, min_target_pixels)
                # unit_positions_elapsed = time.time() - loop_start_time
                # print(f"Unit positions detection lapsed time: {unit_positions_elapsed:.4f}s")
                
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
                        distance_to_previous = ((target_x - prev_x)**2 + (target_y - prev_y)**2)**0.5
 
                        # Predict movement if the new target is within 50 pixels of the previous target location
                        if distance_to_previous < 50:
                            print("predicting...")
                            delta_x = target_x - prev_x
                            delta_y = target_y - prev_y
                            target_x += int(delta_x * 0.4)
                            target_y += int(delta_y * 0.4)
 
                    win32api.SetCursorPos((target_x, target_y))
                    previous_unit_location = (target_x, target_y)
                    # mouse_move_elapsed = time.time() - loop_start_time
                    # print(f"Mouse move lapsed time: {mouse_move_elapsed:.4f}s")
 
                loop__elapsed = time.time() - loop_start_time
                print(f"Loop iteration took {loop__elapsed:.4f} seconds")
                print(f"=================== Found {len(unit_positions)} unit positions ===================")
            else:
                previous_unit_location = None
                time.sleep(0.025) # prevent cpu usage 100%
 
if __name__ == "__main__":
    main()
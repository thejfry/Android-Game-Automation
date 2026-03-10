import time
import numpy as np
import cv2
import random

from framework.device import MyPhone
from framework.vision import Vision
from framework.config import load_config

class AtlasGolfBot:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.phone = MyPhone()
        # self.vision = Vision(self.phone, self.config)

    def detect_indicator_center(self, img):
        """
        Detects the vertical center of the moving indicator.
        Returns (x, y) coordinates or None if not found.
        """
        # 1. Crop the meter region
        region = self.vision.crop_region(img, "indicator_left")

        # 2. Convert to NumPy RGB array
        arr = np.array(region)[..., :3]

        # 3. Load color thresholds
        lower = np.array(self.config["colors"]["indicator"]["lower"])
        upper = np.array(self.config["colors"]["indicator"]["upper"])

        # 4. Create mask of indicator-colored pixels
        mask = np.all((arr >= lower) & (arr <= upper), axis=-1)

        # If no pixels match, return None
        if not mask.any():
            return None

        # 5. Compute centroid
        ys, xs = np.where(mask)
        cx = xs.mean()
        cy = ys.mean()

        # 6. Convert to absolute screen coordinates
        region_x, region_y, _, _ = self.config["regions"]["indicator_left"]
        absolute_x = region_x + cx
        absolute_y = region_y + cy

        return (int(absolute_x), int(absolute_y))

    def render_hud(self, frame, indicator_center, tap, fps):
        # Meter Zone
        x, y, w, h = self.config["regions"]["indicator_left"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        x, y, w, h = self.config["regions"]["four_point_zone"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x, y, w, h = self.config["regions"]["two_point_zone"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)

        # Indicator Center
        if indicator_center is not None:
            cv2.circle(frame, center=indicator_center, radius=6, color=(255, 0, 0), thickness=-1)

        # Text overlays
        self.vision.draw_text_with_bg(frame, f"FPS: {fps:.1f}", 20, 1670)
        self.vision.draw_text_with_bg(frame, f"Tap: {tap}", 20, 1710)
        self.vision.draw_text_with_bg(frame, f"Indicator Detected: {indicator_center}", 20, 1750)

        return frame

    def start(self, render_hud: bool = False):
        print("Connecting to device...")
        self.phone.connect_device()
        print("Connected. Starting Atlas Golf bot...")

        time.sleep(1)
        last_time = time.time()

        indicator_detected = False
        tap = False

        while True:

            img = self.vision.capture_screen()

            indicator_center = self.detect_indicator_center(img)

            if indicator_center is not None:
                indicator_detected = True
                _, indicator_y = indicator_center

                _, four_zone_top, _, four_zone_height = self.config["regions"]["four_point_zone"]
                _, _, _, two_zone_height = self.config["regions"]["two_point_zone"]

                if four_zone_top <= indicator_y <= four_zone_top + four_zone_height + two_zone_height:
                    tap = True
                    # self.phone.tap(self.config["tap_location"]["x"], self.config["tap_location"]["y"])
                else:
                    tap = False
            else:
                indicator_detected = False
                tap = False

            # FPS calculation
            now = time.time()
            fps = 1 / (now - last_time)
            last_time = now

            # Render debugging HUD
            print(f"FPS: {fps:.1f} | Indicator Detected: {indicator_detected} | Tap: {tap}")
            if not render_hud:
                continue

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frame = self.render_hud(
                frame=frame,
                indicator_center=indicator_center,
                tap=tap,
                fps=fps,
            )
            scale = 0.4
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
            cv2.imshow("Atlas Golf Bot HUD", frame_small)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 


    def start_naive_approach(self):
        start_time = time.time()
        print("Connecting to device...")
        self.phone.connect_device()
        print("Connected. Starting Atlas Golf bot...")
        
        now = time.time()
        connection_duration = now - start_time
        print(f"Start Time: {start_time:.1f} | Connection Duration: {connection_duration}")

        # todo: somehow need to detect accurate timing for first tap.
        last_tap = start_time
        time_delays = [10, 9, 6, 4, 3]
        for delay in time_delays:
            time.sleep(delay)
            now = time.time()
            time_since_start = now - start_time
            time_since_last_tap = now - last_tap
            last_tap = now

            self.phone.tap(
                self.config["tap_location"]["x"] + random.randint(-10, 10), 
                self.config["tap_location"]["y"] + random.randint(-10, 10)
            )
            print(f"Time Since Start: {time_since_start:.3f} | Time Since Last Tap: {time_since_last_tap:.3f}")


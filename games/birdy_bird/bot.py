"""
Birdy Bird Bot
Uses:
- device.py for ADB control
- vision.py for screenshot + region analysis
- config.yaml for regions and thresholds
"""

import time
import numpy as np
import cv2

from framework.device import MyPhone
from framework.vision import Vision
from framework.config import load_config


class BirdyBirdBot:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.phone = MyPhone()
        self.vision = Vision(self.phone, self.config)

    # def detect_pipe_gap(self, img):
    #     """
    #     Returns the vertical center of the pipe gap,
    #     or None if no pipe is present.
    #     """
    #     pipe_region = self.vision.crop_region(img, "pipe_scan")
    #
    #     lower = np.array(self.config["colors"]["pipe_green"]["lower"])
    #     upper = np.array(self.config["colors"]["pipe_green"]["upper"])
    #
    #     arr = np.asarray(pipe_region)
    #
    #     # Mask of pipe-colored pixels
    #     mask = np.all((arr >= lower) & (arr <= upper), axis=-1)
    #
    #     # Collapse horizontally → 1D vertical profile
    #     vertical_profile = mask.mean(axis=1)
    #
    #     # If no pipe detected
    #     if vertical_profile.max() < 0.1:
    #         return None
    #
    #     # Pipe = high values, gap = low values
    #     threshold = 0.2
    #     gap_indices = np.where(vertical_profile < threshold)[0]
    #
    #     if len(gap_indices) == 0:
    #         return None
    #
    #     # Find continuous segments of gap
    #     segments = []
    #     start = gap_indices[0]
    #
    #     for i in range(1, len(gap_indices)):
    #         if gap_indices[i] != gap_indices[i - 1] + 1:
    #             segments.append((start, gap_indices[i - 1]))
    #             start = gap_indices[i]
    #
    #     segments.append((start, gap_indices[-1]))
    #
    #     # Choose the largest gap
    #     gap_start, gap_end = max(segments, key=lambda s: s[1] - s[0])
    #     gap_center = (gap_start + gap_end) // 2
    #
    #     return gap_center
    # def detect_pipe_gap(self, img):
    #     pipe_region = self.vision.crop_region(img, "pipe_scan")
    #     arr = np.asarray(pipe_region)
    #
    #     hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    #     h = hsv[..., 0]
    #
    #     mask = (h > 74) & (h < 85)
    #     mask = mask.astype(np.uint8) * 255
    #
    #     kernel = np.ones((15, 1), np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #
    #     vertical_profile = mask.mean(axis=1)
    #
    #     gap_mask = vertical_profile < 10
    #     gap_mask = gap_mask.astype(np.uint8)
    #
    #     num_labels, labels = cv2.connectedComponents(gap_mask)
    #
    #     if num_labels <= 1:
    #         return None
    #
    #     largest = None
    #     largest_size = 0
    #
    #     for i in range(1, num_labels):
    #         ys = np.where(labels == i)[0]
    #         if len(ys) > largest_size:
    #             largest = ys
    #             largest_size = len(ys)
    #
    #     if largest is None:
    #         return None
    #
    #     return int(largest.mean())
    def detect_pipe_gap(self, img):
        """
        Detects only the bottom edge of the top pipe.
        Assumes the pipe gap height is fixed.
        Returns the vertical center of the gap.
        """
        # 1. Crop the upper pipe scan region
        pipe_region = self.vision.crop_region(img, "pipe_scan_top")
        arr = np.asarray(pipe_region)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 3. Vertical edge detection (Sobel Y)
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobel = np.abs(sobel)

        # 4. Collapse horizontally → 1D vertical edge profile
        vertical_profile = sobel.mean(axis=1)

        # 5. Smooth the profile
        vertical_profile = cv2.GaussianBlur(vertical_profile, (9, 1), 0)

        # 6. Find the strongest horizontal edge (bottom of top pipe)
        edge_y = np.argmax(vertical_profile)

        # 7. Convert to absolute screen coordinates
        region_x, region_y, _, _ = self.config["regions"]["pipe_scan_top"]
        top_pipe_bottom_edge = region_y + edge_y

        # 8. Compute gap center using fixed gap height
        gap_height = self.config["gap_height"]
        gap_center = top_pipe_bottom_edge + gap_height // 2

        return gap_center

    def detect_bird_center(self, img):
        """
        Detects the bird's vertical center using color-based centroid tracking.
        Returns the absolute Y coordinate of the bird on the screen,
        or None if the bird cannot be found.
        """
        # 1. Crop the bird search region
        region = self.vision.crop_region(img, "bird_zone")

        # 2. Convert to NumPy RGB array
        arr = np.array(region)[..., :3]

        # 3. Load color thresholds
        lower = np.array(self.config["colors"]["bird_yellow"]["lower"])
        upper = np.array(self.config["colors"]["bird_yellow"]["upper"])

        # 4. Create mask of bird-colored pixels
        mask = np.all((arr >= lower) & (arr <= upper), axis=-1)

        # If no pixels match, return None
        if not mask.any():
            return None

        # 5. Compute centroid (center of mass)
        ys, xs = np.where(mask)
        centroid_y = ys.mean()

        # 6. Convert centroid to absolute screen coordinates
        region_x, region_y, _, _ = self.config["regions"]["bird_zone"]
        absolute_y = region_y + centroid_y

        return int(absolute_y)

    def render_hud(self, frame, bird_center, gap_center, flap, fps):
        # Bird region
        bx, by, bw, bh = self.config["regions"]["bird_zone"]
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

        # Pipe region
        px, py, pw, ph = self.config["regions"]["pipe_scan_top"]
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

        # Bird center
        if bird_center is not None:
            bird_x_center = bx + bw // 2
            cv2.circle(frame, (bird_x_center, bird_center), 6, (0, 255, 255), -1)

        # Pipe gap center
        if gap_center is not None:
            pipe_x_center = px + pw // 2
            cv2.circle(frame, (pipe_x_center, gap_center), 6, (0, 255, 0), -1)

        # Text overlays
        self.vision.draw_text_with_bg(frame, f"Bird Y: {bird_center}", 20, 40)
        self.vision.draw_text_with_bg(frame, f"Gap Y: {gap_center}", 20, 80)

        if bird_center is not None and gap_center is not None:
            error = bird_center - gap_center
            self.vision.draw_text_with_bg(frame, f"Error: {error}", 20, 120)

        self.vision.draw_text_with_bg(frame, f"Flap: {flap}", 20, 160)
        self.vision.draw_text_with_bg(frame, f"FPS: {fps:.1f}", 20, 200)

        return frame

    def start(self, render_hud: bool = False):
        print("Connecting to device...")
        self.phone.connect_device()
        print("Connected. Starting Flappy Bird bot...")

        time.sleep(1)
        last_time = time.time()

        while True:
            img = self.vision.capture_screen()

            # Detect bird and gap heights
            gap_center = self.detect_pipe_gap(img)
            bird_center = self.detect_bird_center(img)

            print(f"Pipe Center: {gap_center} | Bird Center: {bird_center}")

            # Decide whether to flap
            flap = False
            if gap_center is not None and bird_center is not None:
                if bird_center > gap_center + 20:
                    flap = True
                    # self.phone.tap(self.config["tap"]["x"], self.config["tap"]["y"])

            # FPS calculation
            now = time.time()
            fps = 1 / (now - last_time)
            last_time = now

            # Render HUD
            if not render_hud:
                continue

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frame = self.render_hud(
                frame=frame,
                bird_center=bird_center,
                gap_center=gap_center,
                flap=flap,
                fps=fps,
            )
            scale = 0.4
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
            cv2.imshow("BirdyBird Bot HUD", frame_small)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # time.sleep(self.config["loop_delay"])

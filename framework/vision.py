import subprocess
import threading
import numpy as np
import cv2
import time


class Vision:
    def __init__(self, width, height, port=27183, show_debug=False):
        self.width = width
        self.height = height
        self.port = port
        self.show_debug = show_debug

        self.frame_size = width * height * 3
        self.proc = None
        self.thread = None
        self.running = False

        self.frame = None
        self.lock = threading.Lock()

        # HUD overlays
        self.overlays = []
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps = 0

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def start(self):
        """Start FFmpeg and begin reading frames in a background thread."""
        if self.running:
            return

        cmd = [
            "ffmpeg",
            "-f", "h264",
            "-i", f"tcp://127.0.0.1:{self.port}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vsync", "0",
            "-"
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the stream and clean up."""
        self.running = False
        if self.proc:
            self.proc.kill()
            self.proc = None

    def get_frame(self):
        """Return the latest decoded frame (thread-safe)."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def is_running(self):
        return self.running

    # ------------------------------------------------------------
    # HUD overlay API
    # ------------------------------------------------------------

    def draw_rect(self, x, y, w, h, color=(0, 255, 0), thickness=2):
        self.overlays.append(("rect", (x, y, w, h, color, thickness)))

    def draw_circle(self, x, y, radius=5, color=(0, 0, 255), thickness=-1):
        self.overlays.append(("circle", (x, y, radius, color, thickness)))

    def draw_text(self, text, x, y, color=(255, 255, 255), scale=0.6, thickness=1):
        self.overlays.append(("text", (text, x, y, color, scale, thickness)))

    # ------------------------------------------------------------
    # Internal: FFmpeg reader loop
    # ------------------------------------------------------------

    def _reader_loop(self):
        while self.running:
            raw = self.proc.stdout.read(self.frame_size)
            if len(raw) < self.frame_size:
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))

            # FPS calculation
            self.frame_count += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = now

            # Store frame
            with self.lock:
                self.frame = frame

            # Debug HUD
            if self.show_debug:
                self._render_debug(frame.copy())

    # ------------------------------------------------------------
    # Internal: HUD rendering
    # ------------------------------------------------------------

    def _render_debug(self, frame):
        # Draw overlays
        for item in self.overlays:
            kind, data = item

            if kind == "rect":
                x, y, w, h, color, thickness = data
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            elif kind == "circle":
                x, y, radius, color, thickness = data
                cv2.circle(frame, (x, y), radius, color, thickness)

            elif kind == "text":
                text, x, y, color, scale, thickness = data
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, thickness, cv2.LINE_AA)

        # FPS overlay
        cv2.putText(frame, f"FPS: {self.fps}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Vision Debug HUD", frame)
        cv2.waitKey(1)

        # Clear overlays after each frame
        self.overlays.clear()

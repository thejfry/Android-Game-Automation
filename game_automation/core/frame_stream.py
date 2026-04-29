import subprocess
import threading
from queue import Queue
from typing import Optional
import logging
import time
import av
import numpy as np
import os


class FrameStream:
    """
    Streams frames from scrcpy using PyAV to decode H.264 in real time.
    Produces BGR NumPy arrays suitable for OpenCV.
    """

    def __init__(self, max_queue: int = 1, max_size: int = 720, bitrate: int = 8000000):
        self.queue = Queue(maxsize=max_queue)
        self.max_size = max_size
        self.bitrate = bitrate
        self.proc: Optional[subprocess.Popen] = None
        self.scrcpy_proc: Optional[subprocess.Popen] = None
        self.running = False

    def start(self):
        """
        Launch scrcpy and pipe H.264 through ffmpeg.
        """
        # First process: scrcpy outputting H.264
        scrcpy_cmd = [
            "scrcpy",
            "--no-audio",
            "--no-control",
            "--max-size", str(self.max_size),
            "--video-bit-rate", str(self.bitrate)
        ]

        # Second process: ffmpeg wrapping it in MPEGTS
        ffmpeg_cmd = [
            "ffmpeg",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", "pipe:0",
            "-f", "mpegts",
            "-c:v", "copy",
            "-",
            "pipe:1"
        ]

        self.scrcpy_proc = subprocess.Popen(scrcpy_cmd, stdout=subprocess.PIPE)
        self.proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=self.scrcpy_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        self.running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            self.proc = None
        if self.scrcpy_proc:
            self.scrcpy_proc.terminate()
            self.scrcpy_proc = None

    def _reader_loop(self):
        """
        Decode frames from ffmpeg pipe.
        """
        logging.info("Waiting for ffmpeg to initialize...")
        time.sleep(0.5)
        
        try:
            logging.info("Opening ffmpeg stream...")
            container = av.open(self.proc.stdout, format='mpegts')
            logging.info("Container opened successfully")

            for frame in container.decode(video=0):
                if not self.running:
                    break

                img = frame.to_ndarray(format="bgr24")

                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Exception:
                        pass

                self.queue.put(img)

            container.close()
        except Exception as e:
            logging.error(f"Frame stream error: {type(e).__name__}: {e}")
            logging.error(f"scrcpy process returncode: {self.scrcpy_proc.returncode}")
            # Check stderr if available
            if self.scrcpy_proc.stderr:
                err = self.scrcpy_proc.stderr.read()
                if err:
                    logging.error(f"scrcpy stderr: {err}")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Non-blocking fetch of the most recent frame.
        """
        try:
            return self.queue.get_nowait()
        except Exception:
            return None

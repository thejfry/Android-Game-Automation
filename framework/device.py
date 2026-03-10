import adbutils
from typing import Tuple, Sequence
import time


class DeviceNotConnectedError(Exception):
    """Raised when an ADB command is attempted without a connected device."""
    pass

class MyPhone:
    PIN_PAD_NUMBER_POSITIONS = {
            1: (270, 1130),
            2: (550, 1130),
            3: (820, 1130),
            4: (270, 1400),
            5: (550, 1400),
            6: (820, 1400),
            7: (270, 1650),
            8: (550, 1650),
            9: (820, 1650),
            0: (550, 1880),
        }

    def __init__(self):
        self.device = None

    def connect_device(self) -> bool:
        """Connect to the first available ADB device."""
        try:
            self.device = adbutils.adb.device()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ADB: {e}")

        return True

    def _ensure_connected(self):
        if self.device is None:
            raise DeviceNotConenctedError("No device connected. Call connect_device() first")

    def tap(self, x: int, y: int):
        """Simulate a tap at the given coordinates."""
        self._ensure_connected()
        self.device.shell(f"input tap {x} {y}")

    def swipe(self, start: Tuple[int, int], end: Tuple[int, int], duration_ms: int = 300):
        """Simulate a swipe gesture"""
        self._ensure_connected()
        sx, sy = start
        ex, ey = end
        self.device.shell(f"input swipe {sx} {sy} {ex} {ey} {duration_ms}")

    def unlock_lockscreen(self, pin: Sequence[int]):
        """Unlock the phone using a numeric PIN."""
        self._ensure_connected()

        # Wake screen
        self.tap(977, 1800)
        time.sleep(0.2)

        # Swipe up to reveal PIN pad
        self.swipe((820, 1650), (820, 730))
        time.sleep(0.8)

        # Enter PIN digits
        for digit in pin:
            x, y = self.PIN_PAD_NUMBER_POSITIONS[digit]
            self.tap(x, y)
            time.sleep(0.1)

        # Press enter
        self.tap(820, 1880)

import time

from game_automation.core.device import MyPhone
from game_automation.core.frame_stream import FrameStream
from game_automation.core.events import StateBus
from game_automation.core.hud import HUD

from game_automation.games.example_game.perception import ExampleGamePerception
from game_automation.games.example_game.agent import ExampleGameAgent

def main():
    phone = MyPhone()
    phone.connect_device()

    state_bus = StateBus()

    stream = FrameStream(max_queue=1)
    stream.start()

    perception = ExampleGamePerception(state_bus)
    hud = HUD(stream, state_bus, target_hz=20)
    hud.start()

    agent = ExampleGameAgent(phone, state_bus, tick_hz=5)

    try:
        while True:
            frame = stream.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            perception.process(frame)
            # Agent runs in same thread here; you can also thread it.
            agent.tick(state_bus.latest())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        hud.stop()

if __name__ == "__main__":
    main()

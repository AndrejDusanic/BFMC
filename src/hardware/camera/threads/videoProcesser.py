import threading
import time 

class FrameBuffer:
    """Jednostavni buffer sa kapacitetom 1 koji drži frame i timestamp preuzimanja."""
    def __init__(self):
        self.frame = None
        # Kreiramo poseban queue za YOLO detekcije
        self.queuesList["yolo_output"] = threading.Queue()

        self.timestamp = 0
        self.lock = threading.Lock()
        self.event = threading.Event()
     

    def update(self, frame):
        with self.lock:
            self.frame = frame
            self.timestamp = time.time()
            self.event.set()  # signaliziramo da je novi frame dostupan

    def get(self):
        with self.lock:
            return self.frame, self.timestamp

    def clear(self):
        self.event.clear()

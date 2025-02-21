# src/hardware/vehicle/processControl.py

from src.control.pid_controller import PIDController
import time

class processControl:
    def __init__(self, queuesList, logger):
        self.queuesList = queuesList
        self.logger = logger
        self.automatic_mode = False
        self.pid = PIDController(kp=0.5, ki=0.05, kd=0.02)

    def switch_mode(self, mode):
        self.automatic_mode = mode
        self.logger.info(f"Mode switched to {'Automatic' if mode else 'Manual'}")

    def control_loop(self):
        while True:
            if self.automatic_mode:
                error = self.get_lane_error()
                dt = 0.1  # Vrijeme između ciklusa
                steering_angle = self.pid.compute(error, dt)
                self.apply_steering(steering_angle)
            else:
                self.manual_control()

            time.sleep(0.1)

    def get_lane_error(self):
        # Pretpostavka: Greška dolazi iz kamere (izračunata devijacija od centra trake)
        if not self.queuesList["General"].empty():
            lane_data = self.queuesList["General"].get()
            return lane_data.get("lane_error", 0)
        return 0

    def apply_steering(self, angle):
        self.logger.info(f"Applying steering angle: {angle:.2f}")
        # Ovdje se poziva funkcija za upravljanje motorima/servo uređajem

    def manual_control(self):
        # Logika za manuelno upravljanje (npr. preko tastature)
        pass

from enum import auto
import cv2
import numpy as np

class PIDController:
    def __init__(self, kp=250.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0

    def update(self, error, dt):
        """Izračunava PID izlaz na osnovu trenutne greške i proteklog vremena dt."""
        self.integral += error * dt

        derivative = (error - self.last_error) / dt if dt > 0 else 0.0

        output = 0.7*self.kp * error/512 + self.ki * self.integral + self.kd * derivative

        self.last_error = error
        return output

# Globalni objekat PID kontrolera
pid_controller = PIDController()

# Globalna zastavica da li je auto režim aktivan
auto_mode = True

def set_auto_mode(mode: bool):
    """Postavlja globalni auto_mode – kada je True, PID se koristi za automatsko upravljanje."""
    global auto_mode
    if auto_mode:
        auto_mode = True
    else: 
        auto_mode = False
    
#------------------------------



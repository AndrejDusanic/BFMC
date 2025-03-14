import copy
import cv2
import threading
import base64
import picamera2
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil 
from ultralytics import YOLO
import torch
import os
import json
import logging
import traceback

from src.utils.messages.allMessages import (
    mainCamera,
    serialCamera,
    Recording,
    Record,
    Brightness,
    Contrast,
)

from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from .upravljanje import auto_mode, pid_controller, set_auto_mode

import socketio

# Globalne varijable za detekciju znakova i reakcije
sign_reaction_active = False
sign_reaction_complete = threading.Event()
sign_reaction_complete.set()  # Inicijalno postavljeno na True (nema reakcije u toku)

# Globalni SocketIO klijent
sio = socketio.Client()

@sio.event
def connect():
    print("Povezano na PID server.")

@sio.event
def disconnect():
    print("Diskonektovano sa PID server.")

# --- Pomoćne klase ---
class DummyAttr:
    """Pomoćna klasa za atribute poruka."""
    def __init__(self, val):
        self.value = val

class AutoMode:
    """Dummy klasa za AutoMode poruke."""
    Queue = DummyAttr("AutoControl")
    Owner = DummyAttr("AutoControl")
    msgID = DummyAttr("AutoControl")
    msgType = DummyAttr("AutoControl")

class FrameBuffer:
    """Thread-safe bafer frejmova sa kapacitetom od 1."""
    def __init__(self):
        self.frame = None
        self.timestamp = 0
        self.lock = threading.Lock()
        self.event = threading.Event()

    def update(self, frame):
        """Ažurira bafer sa novim frejmom, čisteći prethodni frejm."""
        with self.lock:
            # Postavi novi frejm
            self.frame = frame
            self.timestamp = time.time()
            # Signalizira da je novi frejm dostupan
            self.event.set()

    def get(self):
        """Uzmi trenutni frejm i vremensku oznaku."""
        with self.lock:
            return self.frame, self.timestamp

    def clear(self):
        """Očisti događaj koji označava dostupnost frejma."""
        self.event.clear()

class SignReaction:
    """Klasa koja upravlja reakcijama na različite saobraćajne znakove."""
    def __init__(self, logger):
        self.logger = logger
        self.sign_confidence_threshold = 0.05  # Smanjen prag za osetljiviju detekciju
     
    def reaction_to_0(self, sio):
        #adventage
        """Nema specifične reakcije za znak 0."""
        self.logger.info("REACTION: Sign 0 - No specific action")
        return None
    
    def reaction_to_3(self, sio):
        #no higway
        """Srednja brzina."""
        self.logger.info("REACTION: Sign 3 - Medium speed")
        control_output = {"steer": 0, "speed": 20}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(1)  
    
    def reaction_to_6(self, sio):
        #pedestrian
        """Znak za pjesacki"""
        self.logger.info("REACTION: Sign 6 - Crosswalk")
        control_output = {"steer": 0, "speed": 0}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(4)
        control_output = {"steer": 0, "speed": 10}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(5)  
        
    def reaction_to_1(self, sio):
        #pjesak bike
        """Privremeni stop."""
        self.logger.info("REACTION: Sign 1 - Pedestrian")
        control_output = {"steer": 0, "speed": 0}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(3)  
    
    def reaction_to_4(self, sio):
        #one way sign
        """Nema specifične reakcije za znak 4."""
        self.logger.info("REACTION: Sign 4 - One way sign")
        return None
    
    def reaction_to_7(self, sio):
        #kruzni tok
        """Roundabout."""
        self.logger.info("REACTION: Sign 7 - Roundabout")
        control_output = {"steer": 0, "speed": 0}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(2) 
        control_output = {"steer": 25, "speed": 20}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(1) 
        control_output = {"steer": -25, "speed": 20}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(2) 
        control_output = {"steer": 25, "speed": 20}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(1) 
        
    def reaction_to_2(self, sio):
        #higway
        """Highway."""
        self.logger.info("REACTION: Sign 2 - Highway")
        control_output = {"steer": 0, "speed": 50}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(0.5)
        control_output = {"steer": 10, "speed": 50}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(1)
        control_output = {"steer": -10, "speed": 50}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(1)

    def reaction_to_5(self, sio):
        #parking
        self.logger.info("REACTION: Parking sign detected - Starting forward right side parking maneuver")
        
        # 1. Faza: Prilazak parking prostoru
        control_output = {"steer": 0, "speed": 15}  # Sporija brzina za pristup
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(4)
        
        # 2. Faza: Usporavanje
        control_output = {"steer": 0, "speed": 10}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(2)
        
        # 3. Faza: Početak skretanja desno
        control_output = {"steer": 20, "speed": 10}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(3)  # Skretanje prema parking mestu
        
        # 4. Faza: Jače skretanje da se vozilo poravna sa parking mestom
        control_output = {"steer": 25, "speed": 8}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(3)
        
        # 5. Faza: Izravnavanje volana i kretanje pravo
        control_output = {"steer": 0, "speed": 8}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(1)
        
        # 5. Faza: Izravnavanje volana i kretanje pravo
        control_output = {"steer": 0, "speed": -10}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(4)
        
        # 6. Faza: Finalno zaustavljanje
        control_output = {"steer": 0, "speed": 0}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        self.logger.info("REACTION: Forward parking maneuver completed")
        time.sleep(5)
        
    
    def reaction_to_8(self, sio):
        #stop
        """Potpuni stop."""
        self.logger.info("REACTION: Sign 8 - Stop")
        control_output = {"steer": 0, "speed": 0}
        if sio.connected:
                try:
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
        else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
        time.sleep(3)  
        
               
    def react(self, sign):
        """Reaguj na specifični ID znaka."""
        global sign_reaction_active, sign_reaction_complete
        
        try:
            sign_reaction_active = True
            sign_reaction_complete.clear()
            
            reaction_methods = {
                0: self.reaction_to_0,
                1: self.reaction_to_1,
                2: self.reaction_to_2,
                3: self.reaction_to_3,
                4: self.reaction_to_4,
                5: self.reaction_to_5,
                6: self.reaction_to_6,
                7: self.reaction_to_7,
                8: self.reaction_to_8
            }
            
            # Pozovi odgovarajuću metodu reakcije ako je ID znaka prepoznat
            if sign in reaction_methods:
                self.logger.info(f"Executing reaction for sign {sign}")
                reaction_methods[sign](sio)
                self.logger.info(f"Reaction for sign {sign} completed")
            else:
                self.logger.warning(f"Unknown sign ID: {sign}")
        except Exception as e:
            self.logger.error(f"Error in sign reaction: {e}")
        finally:
            sign_reaction_active = False
            sign_reaction_complete.set()


class threadCamera(ThreadWithStop):
    """Nit koja upravlja funkcionalnošću kamere, detekcijom traka i znakova."""
    def __init__(self, queuesList, logger, debugger):
        super(threadCamera, self).__init__()
        # Inicijalizacija osnovnih atributa
        self.queuesList = queuesList
        self.logger = logger or logging.getLogger("CameraThread")
        self.debugger = debugger
        self.frame_rate = 10  # Fiksan FPS = 10
        self.recording = False
        self.video_writer_lane = None
        self.video_writer_sign = None
        
        self.video_writer_combined = None  # Novi pisač za kombinovani video
        
        # YOLO model za detekciju znakova (lenjo učitavanje)
        self.model_path = "/home/BFMC/newBrain/BFMC/src/hardware/camera/threads/best.pt"
        self.model = None
        
        # Handler za reakciju na znakove
        self.sign_reaction = SignReaction(self.logger)
        
        # Zastavice i brave za obradu znakova
        self.last_detection_time = 0
        self.detection_cooldown = 1.0  # sekunde između reakcija
        self.processing_sign = False
        self.sign_lock = threading.Lock()

        # Bazen niti za paralelne operacije
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Pošiljaoci i primaoci poruka
        self.recordingSender = messageHandlerSender(self.queuesList, Recording)
        self.mainCameraSender = messageHandlerSender(self.queuesList, mainCamera)
        self.serialCameraSender = messageHandlerSender(self.queuesList, serialCamera)
        self.autoControlSender = messageHandlerSender(self.queuesList, AutoMode)

        # Postavi pretplatnike
        self.subscribe()
        
        # Inicijalizuj kameru
        self._init_camera()
        
        # Postavi periodične zadatke
        self.Queue_Sending()
        self.Configs()

        # Buferi frejmova za parne/neparne frejmove
        self.frame_buffer_even = FrameBuffer()  # Za detekciju trake
        self.frame_buffer_odd = FrameBuffer()   # Za detekciju znakova
        
        # Događaji za kontrolu niti
        self._acquisition_running = threading.Event()
        self._acquisition_running.set()
        self._processing_running = threading.Event()
        self._processing_running.set()

        # Varijable za kontrolu vremena
        self.start_time = time.time()
        self.first_emit_done = False
        self.last_emit_time = 0
        self.emit_interval = 0.05  # 50ms minimum između kontrolnih signala

        # Pokreni SocketIO klijent
        self.start_socketio_client()
        
        self.previous_steering = 0
        self.previous_lane_center = None
        self.consecutive_no_lines = 0
        self.max_no_lines_frames = 30
        
        
        # Buffer za čuvanje poslednjeg obrađenog frame-a od svake vrste
        self.last_lane_frame = None
        self.last_sign_frame = None
        
        # Pokreni niti
        self.run()
        
    def start_socketio_client(self):
        """Pokreće SocketIO klijent u posebnoj niti."""
        def client_loop():
            server_address = "http://172.20.10.3:5001"
            retry_count = 0
            max_retries = 5
            retry_delay = 5  # početnih 5 sekundi
            
            while True:
                try:
                    if not sio.connected:
                        self.logger.info(f"Connecting to PID server at {server_address}")
                        sio.connect(server_address)
                        retry_count = 0  # Resetuj broj pokušaja na uspešnu konekciju
                    sio.wait()  # Blokira do diskonektovanja
                except Exception as e:
                    retry_count += 1
                    # Eksponencijalno povećanje čekanja sa ograničenjem
                    current_delay = min(retry_delay * (2 ** (retry_count - 1)), 60)
                    self.logger.error(f"Socket.IO connection error: {e}, retry {retry_count} in {current_delay}s")
                    time.sleep(current_delay)
                    
                    # Nakon nekoliko pokušaja, loguj ozbiljnije upozorenje
                    if retry_count >= max_retries:
                        self.logger.critical(f"Failed to connect to SocketIO server after {max_retries} attempts")
                        # Nastavi pokušavati ali sa fiksnim većim zakašnjenjem
                        time.sleep(30)
                        
        threading.Thread(target=client_loop, daemon=True).start()
    
    def process_frame_lane_detection(self, frame):
        """
        Obrada ulaznog frame-a s fokusom na ROI definisanom trapezom.
        Ako je auto režim aktivan, računa se greška između centra slike i
        prosečnog centra detektovanih linija, a zatim se PID kontroler poziva
        da izračuna korektivni izlaz.
        """
        # Initialize class variables if they don't exist yet
        if not hasattr(self, 'previous_lane_center'):
            self.previous_lane_center = None
        if not hasattr(self, 'consecutive_no_lines'):
            self.consecutive_no_lines = 0
        if not hasattr(self, 'max_no_lines_frames'):
            self.max_no_lines_frames = 30

        if frame is None:
            self.logger.error("Primljen main frame je None.")
            return None, None

        resized_frame = cv2.resize(frame, (1024, 540))
        height, width = resized_frame.shape[:2]

        # Grayscale & blur
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (1,1), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 100,400)
        
        # ROI mask
        mask = np.zeros_like(edges)
        
        #originalni roi
        roi_pts = np.array([
            [0, height],
            [width, height],
            [int(2 * width / 3), int(2 * height / 3)],
            [int(width / 3), int(2 * height / 3)]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [roi_pts], 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        # Hough Transform
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=10, minLineLength=15, maxLineGap=15)

        lane_lines = []
        if lines is not None:
            left_fit, right_fit = [], []
            left_region = width * 0.65
            right_region = width * 0.35

            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 == x2:
                        continue  # Skip vertical lines

                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    if abs(slope) < 0.1:
                        continue  # Skip nearly horizontal lines

                    if slope < 0 and x1 < left_region and x2 < left_region:
                        left_fit.append((slope, y1 - slope * x1))
                    elif slope > 0 and x1 > right_region and x2 > right_region:
                        right_fit.append((slope, y1 - slope * x1))

            def make_points(line):
                slope, intercept = line
                y1 = height
                y2 = int(height * 0.5)
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
                return [[x1, y1, x2, y2]]

            if left_fit:
                left_avg = np.average(left_fit, axis=0)
                lane_lines.append(make_points(left_avg))

            if right_fit:
                right_avg = np.average(right_fit, axis=0)
                lane_lines.append(make_points(right_avg))

        # Calculate lane_center based on lane_lines with improved logic
        left_x, right_x = [], []  # Initialize these for later use in error sanity check

        if not lane_lines:
            # No lines detected - keep previous lane center
            self.consecutive_no_lines += 1
            if self.previous_lane_center is not None and self.consecutive_no_lines < self.max_no_lines_frames:
                # Use previous lane center if available and not lost for too long
                self.logger.info(f"No lines detected, using previous lane center ({self.previous_lane_center})")
                lane_center = self.previous_lane_center
            else:
                # Default to center if no previous reference or lost for too long
                lane_center = width // 2
                self.logger.info("No lines detected for too long or no previous reference, using center")
        else:
        
            # Extract line endpoints from lane_lines
            for line in lane_lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    if slope < -0.3:  # Left line
                        left_x.append(x1)  # Use x1 (bottom of line) instead of x2
                    elif slope > 0.3:  # Right line
                        right_x.append(x1)  # Use x1 (bottom of line) instead of x2

            # Calculate lane center with improved logic for unbalanced detection
            if left_x and right_x:
                # Both lines detected - use weighted average with sanity check
                left_pos = np.median(left_x)
                right_pos = np.median(right_x)
                
                # Sanity check the lane width
                lane_width = right_pos - left_pos
                expected_width = width * 0.5  # Expected width is roughly half the image
                lane_center = int((left_pos + right_pos) / 2)
            elif left_x:
                # Only left line detected - need to be smarter about direction
                left_pos = np.median(left_x)
                steering = +25
                lane_center = width * 0.7  # Force lane center to right
                self.logger.info(f"Left line far left, likely right curve - steering RIGHT")
            elif right_x:
                steering = -25
                lane_center = width * 0.3  # Force lane center to right
                self.logger.info(f"Right line far left, likely left curve - steering LEFT")
            else:
                # No useful lines found
                lane_center = width // 2
            
            # Store the lane center for next frame
            self.consecutive_no_lines = 0
            self.previous_lane_center = lane_center

        # Control logic for auto mode
        control_output = None
        
        if auto_mode:
            image_center = width // 2
            error = lane_center - image_center
            
            # Limit error to reasonable values
            max_reasonable_error = width // 3  # One third of image width
            if abs(error) > max_reasonable_error:
                error = max_reasonable_error * (1 if error > 0 else -1)
                self.logger.info(f"Error capped to {error}")
            
            dt = 0.05
            pid_value = pid_controller.update(error, dt)
            
            # Simplified steering logic with deadzone
            if abs(error) < 110:
                # Small error - go straight
                steering = 0
                #self.logger.info("Small error, using zero steering")
            elif left_x and right_x:
                # Both lines visible - use gentle steering (±15)
                steering = max(min(pid_value, 25), -25)  # Dampen PID to max ±15
                #self.logger.info("Both lines visible, using gentle steering")
            elif left_x and not right_x:
                # Only left line visible - stronger left
                steering = 25
                #self.logger.info("Only left line visible, using full left steering")
            elif right_x and not left_x:
                # Only right line visible - stronger right
                steering = -25
                #self.logger.info("Only right line visible, using full right steering")
            else:
                # Fallback logic for no clear lines
                if abs(pid_value) < 25:
                    steering = 0  # Small PID value - go straight
                else:
                    # Use PID but limit to reasonable values
                    steering = max(min(pid_value, 15), -15)
                #self.logger.info(f"No clear lines, using fallback steering: {steering}")
            
            # Store for next frame
            self.previous_steering = steering
            control_output = {"steer": steering, "speed": 20}

            # Visualization
            cv2.circle(resized_frame, (int(lane_center), height - 10), 5, (0, 0, 255), -1)
            cv2.line(resized_frame, (image_center, height), (image_center, height - 50), (255, 0, 0), 2)
            cv2.putText(resized_frame, f"Error: {error:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(resized_frame, f"PID: {pid_value:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.polylines(resized_frame, [roi_pts], isClosed=True, color=(0, 0, 255), thickness=2)

            # Display line loss counter if active
            if self.consecutive_no_lines > 0:
                cv2.putText(resized_frame, f"No lines: {self.consecutive_no_lines}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                self.logger.debug("Nije pronađena nijedna linija u ROI-ju.")
            cv2.putText(resized_frame, f"CONTROL: {control_output}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            # Line visibility info
            cv2.putText(resized_frame, f"Left: {len(left_x)}, Right: {len(right_x)}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Send control output
            if control_output is not None:
                if not self.first_emit_done:
                    elapsed = time.time() - self.start_time
                    if elapsed < 20:
                        wait_time = 20 - elapsed
                        self.logger.info(f"Čekam {wait_time:.2f} sekundi pre prvog slanja control_output")
                        time.sleep(wait_time)
                        self.first_emit_done = True
            if sio.connected:
                try:
                    # Dodatna provera da sign reakcija nije postala aktivna u međuvremenu
                    if not sign_reaction_active:
                        self.logger.info("Lane detection sending control: %s", control_output)
                        sio.emit('control_output', control_output)
                    else:
                        self.logger.info("Skipping lane control - sign reaction active")
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
            else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")
                
        self.last_lane_frame = resized_frame.copy()

        return resized_frame, control_output

    
    def _draw_lane_visualization(self, frame, lines, lane_center, image_center, error, pid_value, roi_pts, control_output, width=None):
        """
        Nacrtaj elemente vizualizacije detekcije trake na frejm.
        
        Parametri:
        -----------
        frame : numpy.ndarray
            Frejm na koji se crta
        lines : list
            Detektovane linije
        lane_center : int
            Centralna pozicija detektovane trake
        image_center : int
            Centralna pozicija slike
        error : float
            Greška između centra trake i slike
        pid_value : float
            Izlaz PID kontrolera
        roi_pts : numpy.ndarray
            Tačke regiona interesa
        control_output : dict
            Vrednosti kontrolnog izlaza
        width : int, opciono
            Širina frejma (daje se eksplicitno da bi se izbegle greške)
        """
        # Uzmi dimenzije frejma (uvek ponovo izračunaj, čak i ako je širina data)
        height, frame_width = frame.shape[:2]
        
        # Koristi datu širinu ako je dostupna, inače koristi širinu frejma
        if width is None:
            width = frame_width
        
        # Indikator centra trake
        cv2.circle(frame, (int(lane_center), height - 10), 5, (0, 0, 255), -1)
        
        # Referentna linija za centar slike
        cv2.line(frame, (image_center, height), (image_center, height - 50), (255, 0, 0), 2)
        
        # Kontura ROI
        cv2.polylines(frame, [roi_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # Nacrtaj detektovane linije
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tekst statusa
        cv2.putText(frame, f"Error: {error:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"PID: {pid_value:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"CONTROL: {control_output}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Status reakcije na znak
        reaction_status = 'Active' if sign_reaction_active else 'Inactive'
        status_color = (0, 0, 255) if sign_reaction_active else (0, 255, 0)
        cv2.putText(frame, f"Sign Reaction: {reaction_status}", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Indikator tipa frejma - Parni
        try:
            cv2.putText(frame, "LANE DETECTION (EVEN FRAME)", 
                    (width - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        except Exception as e:
            self.logger.error(f"Error in _draw_lane_visualization: {e}")
            cv2.putText(frame, "LANE DETECTION (EVEN FRAME)", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    def process_frame_sign_detection(self, frame):
        """
        Unapređena obrada za detekciju saobraćajnih znakova:
        1. Poboljšana predobrada slike
        2. Fokusiranje samo na relevantna područja
        3. Optimizovana ROI maska
        """
        if frame is None:
            self.logger.error("Received sign detection frame is None.")
            return None, None
            
        # Resize za procesiranje
        resized_frame = cv2.resize(frame, (1024, 540))
        height, width = resized_frame.shape[:2]
        
        # Sačuvaj originalnu sliku za vizualizaciju
        original_frame = resized_frame.copy()
        working_frame = resized_frame.copy()
        
        # 1. PREDOBRADA SLIKE
        # Poboljšanje kontrasta korišćenjem CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(working_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        working_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 2. ROI MASKA
        # Kreiraj ROI masku koja isključuje deo slike gde se ne očekuju znakovi
        # Isključi levi deo slike i desni donji deo
        mask = np.ones_like(working_frame)
        
        # Parametri za ROI
        # Maska za levi deo slike (isključi)
        left_mask_pts = np.array([[(0,0), (int((2/3)*width),0), (int((2/3)*width),height), (0,height)]], dtype=np.int32)
        
        # Maska za donji desni deo (isključi)
        right_bottom_mask_pts = np.array([[(int((2/3)*width),int((2/3)*height)), (width,int((2/3)*height)), (width,height), (int((2/3)*width),height)]], dtype=np.int32)
        
        # Primeni maske (postavi na crno područja koja isključujemo)
        cv2.fillPoly(working_frame, left_mask_pts, (0,0,0))
        cv2.fillPoly(working_frame, right_bottom_mask_pts, (0,0,0))
        
        # Leno učitaj YOLO model pri prvoj upotrebi
        if self.model is None:
            try:
                self.model = self._load_model()
            except Exception as e:
                self.logger.error(f"Failed to load YOLO model: {e}")
                return resized_frame, None
        
        # Pokreni detekciju znakova
        control_output = None
        start_time = time.time()
        
        try:
            # Pokrentri YOLO detekciju
            results = self.model(working_frame)
            processing_time = time.time() - start_time
            
            # Obradi detekcije
            detections = []
            best_detection = None
            best_confidence = 0
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append((x1, y1, x2, y2, conf, cls))
                    
                    # Prati detekciju sa najvišom pouzdanošću
                    if conf > best_confidence:
                        best_confidence = conf
                        best_detection = cls
            
            # Nacrtaj ROI granice na originalnom frejmu
            cv2.polylines(original_frame, [left_mask_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.polylines(original_frame, [right_bottom_mask_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Nacrtaj pravougaonike za detektovane znakove
            for x1, y1, x2, y2, conf, cls in detections:
                # Nacrtaj pravougaonik i oznaku na originalnoj slici (ne na radnoj)
                cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Ispis klase i pouzdanosti
                label = f"Sign {cls}: {conf:.2f}"
                cv2.putText(original_frame, label, 
                        (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Dodaj informacije o obradi
            cv2.putText(original_frame, f"Processing time: {processing_time:.3f}s", 
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Reaguj na detektovani znak ako uslovi dozvoljavaju
            current_time = time.time()
            
            if best_detection is not None and best_confidence >= 0.25:  # Podignuta granica pouzdanosti na 0.25
                self.logger.info(f"Detected sign {best_detection} with confidence {best_confidence:.2f}")
                
                # Dodaj informaciju o detektovanom znaku na frejm bez obzira na reakciju
                cv2.putText(original_frame, f"DETECTED: Sign {best_detection}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Proveri da li možemo da pokrenemo reakciju
                if not sign_reaction_active and current_time - self.last_detection_time >= self.detection_cooldown:
                    with self.sign_lock:
                        if not self.processing_sign:
                            self.processing_sign = True
                            self.last_detection_time = current_time
                            
                            # Pokreni reakciju na znak u posebnoj niti
                            self.logger.info(f"⚠️ TRIGGERING reaction for sign {best_detection}!")
                            threading.Thread(
                                target=self._handle_sign_reaction,
                                args=(best_detection,),
                                daemon=True
                            ).start()
            else:
                if best_detection is not None:
                    cv2.putText(original_frame, f"Low conf: Sign {best_detection} ({best_confidence:.2f})", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 191, 255), 2)
            
            # Indikator tipa frejma
            cv2.putText(original_frame, "SIGN DETECTION", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Status reakcije na znak
            reaction_status = 'ACTIVE' if sign_reaction_active else 'INACTIVE'
            status_color = (0, 0, 255) if sign_reaction_active else (0, 255, 0)
            cv2.putText(original_frame, f"Sign Reaction: {reaction_status}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Sačuvaj poslednji obrađeni sign frejm
            self.last_sign_frame = original_frame.copy()
                
        except Exception as e:
            self.logger.error(f"Error in sign detection: {e}")
            traceback_str = traceback.format_exc()
            self.logger.error(f"Traceback: {traceback_str}")
            
        return original_frame, control_output
    
    def _load_model(self):
        """Učitaj YOLO model sa odgovarajućim podešavanjima."""
        self.logger.info(f"Loading YOLO model from {self.model_path}")
        
        try:
            # Proveri dostupne resurse
            if torch.cuda.is_available():
                self.logger.info("CUDA available, using GPU")
                device = 'cuda:0'
            else:
                self.logger.info("CUDA not available, using CPU")
                device = 'cpu'
                
            # Učitaj model sa odgovarajućim podešavanjima
            model = YOLO(self.model_path)
            model.to(device)
            
            # Podešavanja modela optimizovana za osetljiviju detekciju znakova
            model.conf = 0.1  # Vrlo nizak prag pouzdanosti za povećanje osetljivosti
            model.iou = 0.3   # Niži IOU prag za više detekcija
            model.verbose = False  # Isključi verbose ispis
            
            # Opciono: podesi multi-scale detekciju (može biti sporije)
            # model.args['augment'] = True
            
            self.logger.info("YOLO model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
    def _handle_sign_reaction(self, sign_id):
        """Obradi reakciju na znak u posebnoj niti."""
        global sign_reaction_active
        
        try:
            self.logger.info(f"Starting reaction for sign {sign_id}")
            
            # Direktno postavi globalnu varijablu
            sign_reaction_active = True
            sign_reaction_complete.clear()
            
            # Pozovi odgovarajuću metodu reakcije direktno
            self.sign_reaction.react(sign_id)
            
            self.logger.info(f"Completed reaction for sign {sign_id}")
        except Exception as e:
            self.logger.error(f"Error in sign reaction: {e}")
        finally:
            # Obezbedi čišćenje čak i u slučaju izuzetka
            sign_reaction_active = False
            sign_reaction_complete.set()
            with self.sign_lock:
                self.processing_sign = False

    def subscribe(self):
        """Pretplati se na redove poruka."""
        self.recordSubscriber = messageHandlerSubscriber(self.queuesList, Record, "lastOnly", True)
        self.brightnessSubscriber = messageHandlerSubscriber(self.queuesList, Brightness, "lastOnly", True)
        self.contrastSubscriber = messageHandlerSubscriber(self.queuesList, Contrast, "lastOnly", True)
        self.autoModeSubscriber = messageHandlerSubscriber(self.queuesList, AutoMode, "lastOnly", True)

    def Queue_Sending(self):
        """Periodično šalje status snimanja."""
        self.recordingSender.send(self.recording)
        threading.Timer(1, self.Queue_Sending).start()

    def Configs(self):
        """Obrađuje ažuriranja konfiguracije kamere iz poruka."""
        # Obradi promene osvetljenja
        if self.brightnessSubscriber.isDataInPipe():
            message = self.brightnessSubscriber.receive()
            if self.debugger:
                self.logger.info(f"Brightness message: {message}")
            try:
                brightness = max(0.0, min(1.0, float(message)))
                self.camera.set_controls({
                    "AeEnable": False,
                    "AwbEnable": False,
                    "Brightness": brightness,
                })
            except Exception as e:
                self.logger.error(f"Error setting brightness: {e}")
                
        # Obradi promene kontrasta
        if self.contrastSubscriber.isDataInPipe():
            message = self.contrastSubscriber.receive()
            if self.debugger:
                self.logger.info(f"Contrast message: {message}")
            try:
                contrast = max(0.0, min(32.0, float(message)))
                self.camera.set_controls({
                    "AeEnable": False,
                    "AwbEnable": False,
                    "Contrast": contrast,
                })
            except Exception as e:
                self.logger.error(f"Error setting contrast: {e}")
                
        # Zakaži sledeću proveru
        threading.Timer(1, self.Configs).start()

    def adjust_gamma(self, image, gamma=1.2):
        """Primeni korekciju game na sliku."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def _handle_auto_mode(self):
        """Obrađuje promene auto režima na osnovu primljenih poruka."""
        if not self.autoModeSubscriber.isDataInPipe():
            return
            
        message = self.autoModeSubscriber.receive()
        if message is not None:
            try:
                msg_str = str(message).lower()
                new_mode = msg_str == "true"
                
                # Ažuriraj samo ako se režim promenio
                if new_mode != auto_mode:
                    set_auto_mode(new_mode)
                    self.logger.info(f"Auto mode set to {new_mode}")
            except Exception as e:
                self.logger.error(f"Error processing AutoMode message: {e}")

    def _init_camera(self):
        """Inicijalizuje kameru sa odgovarajućom konfiguracijom."""
        self.logger.info("Initializing camera")
        
        try:
            self.camera = picamera2.Picamera2()
            
            # Kreiraj konfiguraciju sa oba toka visoke i niske rezolucije
            config = self.camera.create_preview_configuration(
                buffer_count=4,  # Koristi više bafera za glatko hvatanje sa višestrukim putanjama obrade
                queue=False,     # Ne redaj frejmove da bi se izbeglo kašnjenje
                main={"format": "RGB888", "size": (1280, 720)},  # Glavni tok visoke rezolucije
                lores={"format": "YUV420", "size": (512, 270)}    # Tok niske rezolucije za bržu obradu
            )
            
            # Konfiguriši i pokreni kameru
            self.camera.configure(config)
            
            # Opciono: postavi dodatne parametre kamere
            self.camera.set_controls({
                "FrameRate": 30,       # FPS = 10
                "AwbEnable": True,     # Auto balans bele boje
                "AeEnable": True,      # Auto ekspozicija
                "AnalogueGain": 1.0,   # Osnovni gain
                #"Sharpness": 2.0       # Malo oštrija slika
            })
            
            self.camera.start()
            
            # Dozvoli kameri da se stabilizuje i podesi
            time.sleep(2)
            
            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.critical(f"Failed to initialize camera: {e}")
            raise
        
    def capture_loop(self):
        """Nit koja kontinuirano hvata frejmove i naizmenično ih šalje između bafera."""
        frame_count = 0
        
        # Parametri za performantno hvatanje
        skip_frames = 0  # Preskoči frejmove ako obrada ne može da održi korak
        max_capture_rate = 10  # Maksimalni FPS za hvatanje (fiksno 10)
        min_frame_time = 1.0 / max_capture_rate
        
        while self._running and self._acquisition_running.is_set():
            loop_start = time.time()
            
            # Preskoči frejmove ako je potrebno
            if skip_frames > 0:
                skip_frames -= 1
                time.sleep(0.001)
                continue
                
            try:
                # Uhvati frejm
                frame = self.camera.capture_array("main", wait=True)
                if frame is None:
                    self.logger.error("Capture loop: frame is None")
                    time.sleep(0.01)
                    continue
                    
                # Odredi da li je ovo parni ili neparni frejm
                is_even_frame = (frame_count % 2 == 0)
                
                # Ažuriraj odgovarajući bafer
                if is_even_frame:
                    # Parni frejmovi idu za detekciju trake
                    self.frame_buffer_even.update(frame)
                    if self.debugger and frame_count % 100 == 0:
                        self.logger.debug(f"Updated even buffer with frame {frame_count}")
                else:
                    # Neparni frejmovi idu za detekciju znakova
                    self.frame_buffer_odd.update(frame)
                    self.frame_buffer_even.update(frame)
                    if self.debugger and frame_count % 100 == 1:
                        self.logger.debug(f"Updated odd buffer with frame {frame_count}")
                
                # Povećaj brojač frejmova
                frame_count += 1
                
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                time.sleep(0.1)  # Duže spavanje u slučaju greške
                continue
                
            # Izračunaj proteklo vreme i podesi dinamiku
            loop_time = time.time() - loop_start
            
            # Ograniči stopu hvatanja ako je potrebno
            if loop_time < min_frame_time:
                # Spavaj da bi se održala konzistentna stopa frejmova
                time.sleep(min_frame_time - loop_time)
            elif loop_time > min_frame_time * 1.5:
                # Presporo, možda je potrebno preskočiti frejmove
                skip_frames = int(loop_time / min_frame_time) - 1
                self.logger.debug(f"Capture falling behind, skipping {skip_frames} frames")
                
            # Periodično prikaži FPS
            if frame_count % 100 == 0:
                self.logger.debug(f"Capture rate: {1.0 / max(loop_time, 0.001):.1f} FPS")

    def _handle_recording(self, is_lane_detection):
        """Obrađuje promene stanja snimanja na osnovu primljenih poruka."""
        try:
            if not self.recordSubscriber.isDataInPipe():
                return
                
            recordRecv = self.recordSubscriber.receive()
            if recordRecv is not None:
                self.logger.info(f"Record command received: {recordRecv}")
                new_state = bool(recordRecv)
                
                # Preduzmi akciju samo ako se stanje promenilo
                if new_state != self.recording:
                    self.recording = new_state
                    
                    # Zaustavi snimanje - za sve pisače
                    if not self.recording:
                        if self.video_writer_lane is not None:
                            self.logger.info("Stopping lane detection recording")
                            self.video_writer_lane.release()
                            self.video_writer_lane = None
                            
                        if self.video_writer_sign is not None:
                            self.logger.info("Stopping sign detection recording")
                            self.video_writer_sign.release()
                            self.video_writer_sign = None
                            
                        if self.video_writer_combined is not None:
                            self.logger.info("Stopping combined view recording")
                            self.video_writer_combined.release()
                            self.video_writer_combined = None
                        
                    # Započni snimanje - kreiraj odgovarajuće pisače
                    elif self.recording:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        
                        # Koristi XVID codec za AVI format - široko kompatibilan
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        
                        # Kreiraj pisače za različite tipove snimaka
                        if is_lane_detection:
                            # Lane detection video - uvek poznate dimenzije
                            if self.video_writer_lane is None:
                                filename_lane = f"lane_detection_{timestamp}.avi"
                                self.logger.info(f"Starting lane detection recording to {filename_lane}")
                                self.video_writer_lane = cv2.VideoWriter(
                                    filename_lane,
                                    fourcc,
                                    10,  # FPS = 10
                                    (1024, 540),
                                )
                            
                            # Napravi test kombinovani frejm da bi dobili dimenzije
                            if self.last_lane_frame is not None and self.last_sign_frame is not None:
                                test_combined = self.create_combined_view()
                                if test_combined is not None and self.video_writer_combined is None:
                                    combined_height, combined_width = test_combined.shape[:2]
                                    filename_combined = f"combined_view_{timestamp}.avi"
                                    self.logger.info(f"Starting combined view recording to {filename_combined} with size {combined_width}x{combined_height}")
                                    self.video_writer_combined = cv2.VideoWriter(
                                        filename_combined,
                                        fourcc,
                                        10,  # FPS = 10
                                        (combined_width, combined_height),
                                    )
                            
                        elif not is_lane_detection and self.video_writer_sign is None:
                            # Sign detection video
                            filename_sign = f"sign_detection_{timestamp}.avi"
                            self.logger.info(f"Starting sign detection recording to {filename_sign}")
                            self.video_writer_sign = cv2.VideoWriter(
                                filename_sign,
                                fourcc,
                                10,  # FPS = 10
                                (1024, 540),
                            )
        except Exception as e:
            self.logger.error(f"Record handling error: {e}")
            self.logger.error(traceback.format_exc())

    def create_combined_view(self):
        """
        Kreira kombinovani pregled lane i sign detekcije, sa jednom slikom iznad druge.
        Gornja slika je detekcija trake, a donja je detekcija znakova.
        """
        if self.last_lane_frame is None or self.last_sign_frame is None:
            # Ako jedan od frejmova nije dostupan, vrati None
            return None
                
        # Dimenzije lane frejma
        lane_height, lane_width = self.last_lane_frame.shape[:2]
        
        # Resize sign frejm da odgovara širini lane frejma
        sign_frame_resized = cv2.resize(self.last_sign_frame, (lane_width, int(lane_height)))
        
        # Kreiraj traku za razdvajanje slika
        separator_height = 4
        separator = np.zeros((separator_height, lane_width, 3), dtype=np.uint8)
        separator[:, :] = (0, 255, 255)  # Žuta linija kao separator
        
        # Dodaj tekst za naslove
        lane_title = np.zeros((30, lane_width, 3), dtype=np.uint8)
        sign_title = np.zeros((30, lane_width, 3), dtype=np.uint8)
        
        cv2.putText(lane_title, "LANE DETECTION", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(sign_title, "SIGN DETECTION", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Spoji sve u jednu sliku
        combined_frame = np.vstack([
            lane_title,
            self.last_lane_frame,
            separator,
            sign_title,
            sign_frame_resized
        ])
        
        return combined_frame
     
    def processing_loop_lane_detection(self):
        """Petlja za obradu detekcije trake koristeći parne frejmove."""
        frame_counter = 0
        last_fps_check = time.time()
        frames_processed = 0
        
        while self._running and self._processing_running.is_set():
            loop_start = time.time()
            
            # Obradi konfiguracione promene
            self._handle_recording(is_lane_detection=True)
            self._handle_auto_mode()

            # Čekaj na frejm sa vremenskim ograničenjem
            if not self.frame_buffer_even.event.wait(timeout=0.1):
                continue
                
            # Uzmi frejm iz bafera
            frame, capture_time = self.frame_buffer_even.get()
            self.frame_buffer_even.clear()
            
            if frame is None:
                continue
            
            # Obradi frejm za detekciju trake
            proc_start = time.time()
            processed_frame, control = self.process_frame_lane_detection(frame)
            proc_time = time.time() - proc_start
            
            if processed_frame is None:
                self.logger.error("Lane detection processing failed")
                continue
                
            # Snimaj ako je aktivno
            if self.recording and self.video_writer_lane is not None:
                try:
                    self.video_writer_lane.write(processed_frame)
                except Exception as e:
                    self.logger.error(f"Error writing to lane detection video: {e}")
            
            # Pokušaj kreirati kombinovani prikaz
            combined_frame = self.create_combined_view()
            
            # Ako je kreiranje kombinovanog prikaza uspelo, koristi njega za slanje
            # U suprotnom, koristi samo lane detekciju
            frame_to_send = combined_frame if combined_frame is not None else processed_frame
            
            # Snimaj kombinovani prikaz ako je dostupan i ako je snimanje aktivno
            if self.recording and combined_frame is not None and self.video_writer_combined is not None:
                try:
                    # Samo proveri da li je writer propisno inicijalizovan
                    if self.video_writer_combined.isOpened():
                        self.video_writer_combined.write(combined_frame)
                    else:
                        self.logger.error("Combined video writer not properly initialized")
                except Exception as e:
                    self.logger.error(f"Error writing to combined video: {e}")
                    self.logger.error(traceback.format_exc())
                    
            # Kodiraj i pošalji frejm na frontend
            try:
                _, encoded_img = cv2.imencode(".jpg", frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 80])
                encoded_data = base64.b64encode(encoded_img).decode("utf-8")
                self.mainCameraSender.send(encoded_data)
                # Pošalji na serijsku kameru ali samo svaki drugi frejm da ne bi preopteretio
                #if frame_counter % 2 == 0:
                self.serialCameraSender.send(encoded_data)
            except Exception as e:
                self.logger.error(f"Error encoding/sending frame: {e}")
            
            # Povećaj brojače
            frame_counter += 1
            frames_processed += 1
            
            # Izračunaj i loguj FPS svake sekunde
            current_time = time.time()
            if current_time - last_fps_check >= 1.0:
                fps = frames_processed / (current_time - last_fps_check)
                self.logger.debug(f"Lane detection FPS: {fps:.1f}, Processing time: {proc_time*1000:.1f}ms")
                frames_processed = 0
                last_fps_check = current_time
               
    def processing_loop_sign_detection(self):
        """Petlja za obradu detekcije znakova koristeći neparne frejmove."""
        frame_counter = 0
        last_fps_check = time.time()
        frames_processed = 0
        
        while self._running and self._processing_running.is_set():
            loop_start = time.time()
            
            # Čekaj na frejm sa vremenskim ograničenjem
            if not self.frame_buffer_odd.event.wait(timeout=0.1):
                continue
                
            # Uzmi frejm iz bafera
            frame, capture_time = self.frame_buffer_odd.get()
            self.frame_buffer_odd.clear()
            
            if frame is None:
                continue
            
            # Obradi frejm za detekciju znakova
            proc_start = time.time()
            processed_frame, control = self.process_frame_sign_detection(frame)
            proc_time = time.time() - proc_start
            
            if processed_frame is None:
                self.logger.error("Sign detection processing failed")
                continue
                
            # Snimaj ako je aktivno - samo kao backup jer glavno snimanje ide kroz lane detekciju
            if self.recording and self.video_writer_sign is not None:
                try:
                    self.video_writer_sign.write(processed_frame)
                except Exception as e:
                    self.logger.error(f"Error writing to sign detection video: {e}")
            
            # Povećaj brojače
            frame_counter += 1
            frames_processed += 1
            
            # Izračunaj i loguj FPS svake sekunde
            current_time = time.time()
            if current_time - last_fps_check >= 1.0:
                fps = frames_processed / (current_time - last_fps_check)
                self.logger.debug(f"Sign detection FPS: {fps:.1f}, Processing time: {proc_time*1000:.1f}ms")
                frames_processed = 0
                last_fps_check = current_time
    
    def run(self):
        """Glavna metoda koja pokreće sve potrebne niti za obradu."""
        
        self.logger.info("Starting camera thread with alternating frame processing")
        
        # Inicijalno postavi na None da bi izbegli probleme sa kombinovanim prikazom
        self.last_lane_frame = None
        self.last_sign_frame = None
        
        # Pokreni nit za hvatanje frejmova
        self.capture_thread = threading.Thread(target=self.capture_loop, name="CameraCapture")
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Pokreni nit za detekciju trake (obrađuje parne frejmove)
        self.lane_detection_thread = threading.Thread(target=self.processing_loop_lane_detection, name="LaneDetection")
        self.lane_detection_thread.daemon = True
        self.lane_detection_thread.start()
        
        # Pokreni nit za detekciju znakova (obrađuje neparne frejmove)
        self.sign_detection_thread = threading.Thread(target=self.processing_loop_sign_detection, name="SignDetection")
        self.sign_detection_thread.daemon = True
        self.sign_detection_thread.start()
        
        # Pokušaj postaviti prioritete niti i CPU afinitet ako je moguće
        try:
            # Postavi CPU afinitet za distribuiranje opterećenja po jezgrima
            if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
                capture_cores = {0}      # Koristi jezgro 0 za hvatanje
                lane_cores = {1}         # Koristi jezgro 1 za detekciju trake
                sign_cores = {2, 3}      # Koristi jezgra 2,3 za detekciju znakova (zahtevnija)
                
                os.sched_setaffinity(self.capture_thread.ident, capture_cores)
                os.sched_setaffinity(self.lane_detection_thread.ident, lane_cores)
                os.sched_setaffinity(self.sign_detection_thread.ident, sign_cores)
                
                self.logger.info(f"Set thread affinities: capture={capture_cores}, lane={lane_cores}, sign={sign_cores}")
            
            # Postavi prioritet niti ako je dostupno
            if hasattr(os, 'setpriority') and hasattr(os, 'PRIO_PROCESS'):
                os.setpriority(os.PRIO_PROCESS, self.capture_thread.ident, -5)       # Viši prioritet za hvatanje
                os.setpriority(os.PRIO_PROCESS, self.lane_detection_thread.ident, -2) # Srednji za detekciju trake
                os.setpriority(os.PRIO_PROCESS, self.sign_detection_thread.ident, 0)  # Normalni za detekciju znakova
                
                self.logger.info("Set thread priorities")
                
        except Exception as e:
            self.logger.warning(f"Failed to set thread priorities: {e}")
        
        # Glavna nit nadgleda radničke niti
        while self._running:
            # Proveri zdravlje niti
            thread_status = []
            
            if not self.capture_thread.is_alive():
                thread_status.append("Capture thread died")
                self.logger.error("Capture thread died, restarting")
                self.capture_thread = threading.Thread(target=self.capture_loop, name="CameraCapture")
                self.capture_thread.daemon = True
                self.capture_thread.start()
                
            if not self.lane_detection_thread.is_alive():
                thread_status.append("Lane detection thread died")
                self.logger.error("Lane detection thread died, restarting")
                self.lane_detection_thread = threading.Thread(target=self.processing_loop_lane_detection, name="LaneDetection")
                self.lane_detection_thread.daemon = True
                self.lane_detection_thread.start()
                
            if not self.sign_detection_thread.is_alive():
                thread_status.append("Sign detection thread died")
                self.logger.error("Sign detection thread died, restarting")
                self.sign_detection_thread = threading.Thread(target=self.processing_loop_sign_detection, name="SignDetection")
                self.sign_detection_thread.daemon = True
                self.sign_detection_thread.start()
            
            if thread_status:
                self.logger.warning(f"Thread issues detected: {', '.join(thread_status)}")
                
            # Proveri sistemske resurse
            try:
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 90:
                    self.logger.warning(f"High memory usage: {memory_percent}%")
            except:
                pass
                
            time.sleep(1.0)
            
        # Čišćenje prilikom zaustavljanja
        self._acquisition_running.clear()
        self._processing_running.clear()
        
        # Čekaj da se niti zaustave
        self.logger.info("Waiting for threads to stop")
        self.capture_thread.join(timeout=2.0)
        self.lane_detection_thread.join(timeout=2.0)
        self.sign_detection_thread.join(timeout=2.0)
        
        self.logger.info("Camera thread stopped")
    
    def stop(self):
        """Zaustavi sve niti i očisti resurse."""
        self.logger.info("Stopping camera thread")
        
        if self.video_writer_combined is not None:
            self.logger.info("Stopping combined view recording")
            self.video_writer_combined.release()
            self.video_writer_combined = None
        
        # Zaustavi snimanje ako je aktivno
        if self.recording:
            if self.video_writer_lane is not None:
                self.logger.info("Stopping lane detection recording")
                self.video_writer_lane.release()
                self.video_writer_lane = None
                
            if self.video_writer_sign is not None:
                self.logger.info("Stopping sign detection recording")
                self.video_writer_sign.release()
                self.video_writer_sign = None
            
        # Prekini SocketIO vezu ako je povezan
        if sio.connected:
            try:
                sio.disconnect()
            except:
                pass
                
        # Zaustavi resurse osnovne klase
        super(threadCamera, self).stop()
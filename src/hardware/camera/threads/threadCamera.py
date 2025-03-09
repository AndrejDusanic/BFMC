# src/hardware/camera/threads/threadCamera.py

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
sio = socketio.Client()

@sio.event
def connect():
    print("Povezano na PID server.")

@sio.event
def disconnect():
    print("Diskonektovano sa PID server.")


# --- Dummy poruka za AutoMode ---
# messageHandlerSubscriber očekuje objekat koji ima atribut Owner.
# Pomoćna klasa koja enkapsulira vrednost
class DummyAttr:
    def __init__(self, val):
        self.value = val

class AutoMode:
    # Svi atributi (Queue, Owner, msgID) su instanca DummyAttr s odgovarajućom vrednošću
    Queue = DummyAttr("AutoControl")
    Owner = DummyAttr("AutoControl")
    msgID = DummyAttr("AutoControl")
    msgType = DummyAttr("AutoControl")
# ----------------------------------


class FrameBuffer:
    """ Jednostavni buffer sa kapacitetom 1 koji drži frame i timestamp preuzimanja."""
    def __init__(self):
        self.frame = None
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
        
class threadCamera(ThreadWithStop):
    """Thread koji rukuje funkcionalnostima kamere."""
    def __init__(self, queuesList, logger, debugger):
        super(threadCamera, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_rate = 7  # FPS
        self.recording = False
        self.video_writer = None

        # Inicijalizacija executor-a za paralelnu obradu
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.recordingSender = messageHandlerSender(self.queuesList, Recording)
        self.mainCameraSender = messageHandlerSender(self.queuesList, mainCamera)
        self.serialCameraSender = messageHandlerSender(self.queuesList, serialCamera)
        
        # Sender za automatsko upravljanje (PID izlaz)
        self.autoControlSender = messageHandlerSender(self.queuesList, AutoMode)

        self.subscribe()
        self._init_camera()
        self.Queue_Sending()
        self.Configs()

        # Kreiramo zajednički buffer za frame-ove
        self.frame_buffer_odd = FrameBuffer()
        self.frame_buffer_even = FrameBuffer()
        self._acquisition_running = threading.Event()
        self._acquisition_running.set()
        self._processing_running = threading.Event()
        self._processing_running.set()

        self.start_time = time.time()
        self.first_emit_done = False

        self.start_socketio_client()

    def start_socketio_client(self):
        def client_loop():
            while True:
                try:
                    # Poveži se na server – uveri se da je adresa tačna
                    sio.connect("http://172.20.10.3:5001")
                    sio.wait()  # Ova funkcija blokira i održava konekciju aktivnom
                except Exception as e:
                    self.logger.error("Socket.IO connection error: %s", e)
                    time.sleep(5)  # Pokušaj ponovo nakon 5 sekundi
        threading.Thread(target=client_loop, daemon=True).start()
    
    def process_frame_lane_detection(self, frame):
        """
        Obrada ulaznog frame-a s fokusom na ROI definisanom trapezom.
        Ako je auto režim aktivan, računa se greška između centra slike i
        prosečnog centra detektovanih linija, a zatim se PID kontroler poziva
        da izračuna korektivni izlaz.
        """
        if frame is None:
            self.logger.error("Primljen main frame je None.")
            return None

        resized_frame = cv2.resize(frame, (1024, 540))
        height, width = resized_frame.shape[:2]

        # Grayscale & blur
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 200, 400)

        # ROI mask
        mask = np.zeros_like(edges)
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

        def compute_lane_center(lane_lines):
            """
            Izračunavanje centra trake na osnovu detektovanih linija.
            """
            left_x, right_x = [], []
            for line in lane_lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    if slope < -0.3:
                        left_x.append(x2)
                    elif slope > 0.3:
                        right_x.append(x2)

            if left_x and right_x:
                return (np.median(left_x) + np.median(right_x)) // 2
            elif left_x:
                return np.median(left_x) + (width * 0.25)
            elif right_x:
                return np.median(right_x) - (width * 0.25)
            else:
                return width // 2


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

        lane_center = compute_lane_center(lane_lines)

        # Ako je auto režim aktivan, obračunavamo grešku i koristimo PID kontroler
        control_output = None
        
        if auto_mode:
            image_center = width // 2
            error = lane_center - image_center
            dt = 0.05  # Pretpostavljena vremenska razlika
            pid_value = pid_controller.update(error, dt)
            steering = max(min(pid_value, 25), -25)
            control_output = {"steer": steering, "speed": 20}

            # Dodavanje informacija na sliku
            cv2.circle(resized_frame, (int(lane_center), height - 10), 5, (0, 0, 255), -1)
            cv2.line(resized_frame, (image_center, height), (image_center, height - 50), (255, 0, 0), 2)
            cv2.putText(resized_frame, f"Error: {error:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(resized_frame, f"PID: {pid_value:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.polylines(resized_frame, [roi_pts], isClosed=True, color=(0, 0, 255), thickness=2)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                self.logger.debug("Nije pronađena nijedna linija u ROI-ju.")
            cv2.putText(resized_frame, f"CONTROL: {control_output}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
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
                    sio.emit('control_output', control_output)
                except Exception as e:
                    self.logger.error("Greška pri slanju control_output: %s", e)
            # Nakon uspešnog slanja izlazimo iz petlje
            else:
                self.logger.error("Socket.IO nije konektovan, preskačem slanje")

        return resized_frame, control_output

    def process_frame_sign_detection(self, frame):
        
        def detect_signs(frame):
            """Pokreće YOLO model i vraća listu detekcija."""
            model_path="/home/BFMC/newBrain/BFMC/src/hardware/camera/threads/best.pt"
            model = YOLO(model_path)
            results = model(frame)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append((x1, y1, x2, y2, conf, cls))
            return detections

        if frame is None:
            self.logger.error("Primljen main frame je None.")
            return None
        
        # Čekamo na novi frame u bufferu (timeout 10ms)
        while not self.frame_buffer_even.event.wait(timeout=0.01):
            continue
            
        frame_output, timestamp = self.frame_buffer_even.get()
        self.frame_buffer_even.clear()

        while frame_output is None:
            continue

        # Define the region to crop (y1:y2, x1:x2)
        x1, y1, x2, y2 = 682, 0, 1024, 540  # Adjust coordinates as needed
        cropped_image = frame[:, int((2*1024)//3):1024]  # Slicing

        frame_output =  cropped_image    
        control_output = None

        start_time = time.time()
        detections = detect_signs(frame_output)
        end_time = time.time()

        processing_time = end_time - start_time
        self.logger.info(f"YOLOv8 processing time: {processing_time:.3f}s, Detections: {len(detections)}")

        two_thirds = (1024 * 2) // 3
        # Crtanje bounding boxova
        for x1, y1, x2, y2, conf, cls in detections:
            
            cv2.rectangle(frame, (two_thirds + int(x1), int(y1)), (two_thirds + int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (two_thirds + int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"detected: {cls}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"time:__{processing_time}s__", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        return frame, control_output

    def subscribe(self):
        """Pretplate na poruke (record, brightness, contrast, auto mode)."""
        self.recordSubscriber = messageHandlerSubscriber(self.queuesList, Record, "lastOnly", True)
        self.brightnessSubscriber = messageHandlerSubscriber(self.queuesList, Brightness, "lastOnly", True)
        self.contrastSubscriber = messageHandlerSubscriber(self.queuesList, Contrast, "lastOnly", True)
        # Pretplata na AutoMode poruke pomoću dummy klase AutoMode
        self.autoModeSubscriber = messageHandlerSubscriber(self.queuesList, AutoMode, "lastOnly", True)

    def Queue_Sending(self):
        """Periodično slanje statusa snimanja."""
        self.recordingSender.send(self.recording)
        threading.Timer(1, self.Queue_Sending).start()

    def Configs(self):
        """Podešavanje kamera parametara (brightness, contrast) na osnovu primljenih poruka."""
        if self.brightnessSubscriber.isDataInPipe():
            message = self.brightnessSubscriber.receive()

            if self.debugger:
                self.logger.info(f"Brightness poruka: {message}")
            self.camera.set_controls({
                "AeEnable": False,
                "AwbEnable": False,
                "Brightness": max(0.0, min(1.0, float(message))),
            })
        if self.contrastSubscriber.isDataInPipe():
            message = self.contrastSubscriber.receive()
            if self.debugger:
                self.logger.info(f"Contrast poruka: {message}")
            self.camera.set_controls({
                "AeEnable": False,
                "AwbEnable": False,
                "Contrast": max(0.0, min(32.0, float(message))),
            })
        threading.Timer(1, self.Configs).start()

    def adjust_gamma(self, image, gamma=1.2):
        """Primenjuje gamma korekciju na ulaznu sliku."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def capture_loop(self):
        """Nit koja kontinuirano preuzima frame-ove i ažurira shared buffer."""
        counter = 0
        while self._running and self._acquisition_running.is_set():
            if counter // 2 == 0:
                try:
                    frame = self.camera.capture_array("main", wait=True)
                    if frame is not None:
                        self.frame_buffer_odd.update(frame)                   #koristimo ga za lane detection
                    else:
                        self.logger.error("Capture loop: frame je None.")
                except Exception as e:
                    self.logger.error(f"Capture loop error: {e}")
                time.sleep(0.001)
            
            if counter // 2 == 1:
                try:
                    frame = self.camera.capture_array("main", wait=True)
                    if frame is not None:
                        self.frame_buffer_even.update(frame)                   #koristimo ga za yolo model
                    else:
                        self.logger.error("Capture loop: frame je None.")
                except Exception as e:
                    self.logger.error(f"Capture loop error: {e}")
                time.sleep(0.001)
            counter += 1
            if counter == 100:
                counter = 0

    def processing_loop_lane_detection(self):
        """Nit koja čeka frame u bufferu, obrađuje ga, meri kašnjenje i, ako je snimanje aktivno, zapisuje ga u video."""
        USE_FIXED_COMMAND = True  # Postavite na True da testirate fiksnu komandu; False da se koristi PID izlaz
        while self._running and self._processing_running.is_set():
            # Provera poruka za AutoMode (npr. "true"/"false")
            if self.autoModeSubscriber.isDataInPipe():
                message = self.autoModeSubscriber.receive()
                # self.logger.info("AutoMode message received: %s", message)
                if message is not None:
                    try:
                        msg_str = str(message).lower()
                        if msg_str == "true":
                            set_auto_mode(True)
                            self.logger.info("Auto mode set to True")
                        else:
                            set_auto_mode(False)
                            self.logger.info("Auto mode set to False")
                    except Exception as e:
                        self.logger.error("Error processing AutoMode message: %s", e)

            try:
                recordRecv = self.recordSubscriber.receive()
                if recordRecv is not None:
                    self.logger.info(f"Record command received: {recordRecv}")
                    new_state = bool(recordRecv)
                    if new_state != self.recording:
                        self.recording = new_state
                        if not self.recording and self.video_writer is not None:
                            self.logger.info("Stopping recording; releasing video writer")
                            self.video_writer.release()
                            self.video_writer = None
                        elif self.recording and self.video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"XVID")
                            filename = "output_video_" + str(time.time()) + ".avi"
                            self.logger.info(f"Starting recording; initializing video writer with filename {filename}")
                            self.video_writer = cv2.VideoWriter(
                                filename,
                                fourcc,
                                self.frame_rate,
                                (1024, 540),
                            )
            except Exception as e:
                self.logger.error(f"Record subscriber error: {e}")

            if not self.frame_buffer_odd.event.wait(timeout=0.005):
                continue
            frame, capture_time = self.frame_buffer_odd.get()
            self.frame_buffer_odd.clear()
            if frame is None:
                continue
            proc_start = time.time()
            processed_frame, control = self.process_frame_lane_detection(frame)
            proc_end = time.time()

            if processed_frame is None:
                self.logger.error("Obrada frame-a nije uspjela.")
                continue
            
            if self.recording and self.video_writer is not None:
                self.video_writer.write(processed_frame)
                
            _, encodedImg = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encodedImageData = base64.b64encode(encodedImg).decode("utf-8")
            self.mainCameraSender.send(encodedImageData)
            self.serialCameraSender.send(encodedImageData)

            # Slanje automatske upravljačke komande (PID izlaz) ka front-endu
            if control is not None:
                if USE_FIXED_COMMAND:
                    # Koristimo fiksnu komandu: upravljanje sa nulom na volanu i brzinom 50
                    control = {"steer": 0, "speed": 5}
                    # self.logger.info("Using fixed command: %s", control)
                else:
                    self.logger.info("Computed auto command: %s", control)
                auto_control_message = {"channel": "AutoControl", "data": control}
                self.autoControlSender.send(json.dumps(auto_control_message))
            # Bez dodatnog spavanja

    def processing_loop_sign_detection(self):
        """Nit koja čeka frame u bufferu, obrađuje ga, meri kašnjenje i, ako je snimanje aktivno, zapisuje ga u video."""
        USE_FIXED_COMMAND = True  # Postavite na True da testirate fiksnu komandu; False da se koristi PID izlaz
        while self._running and self._processing_running.is_set():
            # Provera poruka za AutoMode (npr. "true"/"false")
            if self.autoModeSubscriber.isDataInPipe():
                message = self.autoModeSubscriber.receive()
                # self.logger.info("AutoMode message received: %s", message)
                if message is not None:
                    try:
                        msg_str = str(message).lower()
                        if msg_str == "true":
                            set_auto_mode(True)
                            self.logger.info("Auto mode set to True")
                        else:
                            set_auto_mode(False)
                            self.logger.info("Auto mode set to False")
                    except Exception as e:
                        self.logger.error("Error processing AutoMode message: %s", e)

            try:
                recordRecv = self.recordSubscriber.receive()
                if recordRecv is not None:
                    self.logger.info(f"Record command received: {recordRecv}")
                    new_state = bool(recordRecv)
                    if new_state != self.recording:
                        self.recording = new_state
                        if not self.recording and self.video_writer is not None:
                            self.logger.info("Stopping recording; releasing video writer")
                            self.video_writer.release()
                            self.video_writer = None
                        elif self.recording and self.video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"XVID")
                            filename = "output_video_" + str(time.time()) + ".avi"
                            self.logger.info(f"Starting recording; initializing video writer with filename {filename}")
                            self.video_writer = cv2.VideoWriter(
                                filename,
                                fourcc,
                                self.frame_rate,
                                (1024, 540),
                            )
            except Exception as e:
                self.logger.error(f"Record subscriber error: {e}")

            if not self.frame_buffer_even.event.wait(timeout=0.05):
                continue
            
            frame, capture_time = self.frame_buffer_even.get()
            self.frame_buffer_even.clear()
            
            if frame is None:
                continue
            
            proc_start = time.time()
            processed_frame, control = self.process_frame_sign_detection(frame)
            proc_end = time.time()

            if processed_frame is None:
                self.logger.error("Obrada frame-a nije uspjela.")
                continue
            
            if self.recording and self.video_writer is not None:
                self.video_writer.write(processed_frame)
            
            _, encodedImg = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encodedImageData = base64.b64encode(encodedImg).decode("utf-8")
            self.mainCameraSender.send(encodedImageData)
            self.serialCameraSender.send(encodedImageData)


    def run(self):
        """Glavna metoda koja pokreće acquisition i processing niti."""
        self.capture_thread = threading.Thread(target=self.capture_loop, name="CameraCapture")
        #self.processing_loop_lane_detection_thread = threading.Thread(target=self.processing_loop_lane_detection, name="CameraProcessingForLines")
        self.processing_loop_sign_detection_thread = threading.Thread(target=self.processing_loop_sign_detection, name="CameraProcessingForSigns")
        
        self.capture_thread.start()
        #self.processing_loop_lane_detection_thread.start()
        self.processing_loop_sign_detection_thread.start()
        
        while self._running:
            time.sleep(0.1)
        self._acquisition_running.clear()
        self._processing_running.clear()
        self.capture_thread.join()
        #self.processing_loop_lane_detection_thread.join()
        self.processing_loop_sign_detection_thread.join()
        
    def stop(self):
        """Zaustavlja nit; ukoliko se snima, oslobađa VideoWriter."""
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        super(threadCamera, self).stop()

    def start(self):
        super(threadCamera, self).start()

    def _init_camera(self):
        """Inicijalizuje objekat kamere sa dva kanala: 'main' i 'lores'."""
        self.camera = picamera2.Picamera2()
        config = self.camera.create_preview_configuration(
            buffer_count=1,
            queue=False,
            main={"format": "RGB888", "size": (2048, 1080)},
            lores={"format": "YUV420", "size": (512, 270)}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)

if __name__ == "__main__":
    # Primer pokretanja kamere u debug režimu
    cam = threadCamera({}, None, True)

    cam.stop()
    cv2.destroyAllWindows()

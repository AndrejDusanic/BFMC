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


def filter_points(x_tmp):
    if len(x_tmp) < 3:
        return None
    x_left = 0
    x_right = 0
    img_center = 512
    x_left_points = []
    x_right_points = []
    new_tmp_left = -1
    new_tmp_right = -1
    ret_tmp = []
    for x in x_tmp:
        if x < img_center:
            x_left+=1
            x_left_points.append(x)
        else:
            x_right+=1
            x_right_points.append(x)
    
    if x_left > 1:
        new_tmp_left = np.mean(x_left_points)
    else: 
        if len(x_left_points)!=0:
            new_tmp_left = x_left_points[0]
    
    if x_right > 1:
        new_tmp_right = np.mean(x_right_points)
    else:
        if len(x_right_points)!=0:
            new_tmp_right = x_right_points[0]
    #print(len(x_right_points), len(x_left_points))
    
    ret_tmp.append(round(new_tmp_left))
    ret_tmp.append(round(new_tmp_right))

    return ret_tmp


def road_center(x_tmp):
    if x_tmp is None or len(x_tmp) == 0:
        return 512 
    x_cur = None
    if len(x_tmp) == 2:
            x_cur = x_tmp[0] + (x_tmp[1] - x_tmp[0])/2
    elif len(x_tmp) == 1:
        if (x_tmp[0] > 720):
            x_cur = 928
        else:
            x_cur = x_tmp[0] 
    return x_cur

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
        self.frame_buffer = FrameBuffer()
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
    
    

    
    def process_frame(self, frame):
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
            error = image_center - lane_center
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
            
            
            

        return resized_frame, control_output
  
    def process_frame_ZUTELINIJE(self, frame):
        """
        Obrada ulaznog frame‑a s fokusom na ROI definisanom trapezom.
        Implementira detekciju saobraćajnih traka koristeći Canny edge detection i Hough transform.
        Računa grešku između centra slike i prosečnog centra detektovanih linija za PID kontroler.
        """
        if frame is None:
            self.logger.error("Primljen main frame je None.")
            return None

        # Resize frame (kao u originalnoj implementaciji)
        resized_frame = cv2.resize(frame, (1024, 540))
        height, width = resized_frame.shape[:2]

        # Preprocessing slike (slično kao u prvom kodu)
        # Convert to grayscale
        gray_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
        # Remove noise with Gaussian blur
        denoised_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
        # Canny edge detection
        edges_image = cv2.Canny(denoised_image, 200, 400)

        # Region of Interest - definisanje trapezoidne oblasti
        mask = np.zeros_like(edges_image)
        vertices = np.array([[
            (width * 0.10, height * 0.70),
            (width * 0.90, height * 0.70),
            (width * 0.95, height * 0.95),
            (width * 0.05, height * 0.95),
        ]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, 255)
        roi_image = cv2.bitwise_and(edges_image, mask)

        # Hough Transform za detekciju linija
        lines = cv2.HoughLinesP(roi_image, 1, np.pi/180, 
                                threshold=10, 
                                minLineLength=15, 
                                maxLineGap=15)

        lane_lines = []
        lines_kept = []

        if lines is not None:
            left_fit, right_fit = [], []
        
            # Definisanje granica za levu i desnu stranu
            boundary_left = 0.35
            boundary_right = 0.4
            left_region_boundary = width * (1 - boundary_left)
            right_region_boundary = width * boundary_right

            for line_segment in lines:
                for x1, y1, x2, y2 in line_segment:
                    if x1 == x2:  # skip vertical lines
                        continue

                    fit = np.polyfit((x1, x2), (y1, y2), 1)
                    slope, intercept = fit[0], fit[1]

                    if abs(slope) < 0.1:  # skip horizontal lines
                        continue

                    lines_kept.append(line_segment)
                
                    if slope < 0:
                        if x1 < left_region_boundary and x2 < left_region_boundary:
                            left_fit.append((slope, intercept))
                    else:
                        if x1 > right_region_boundary and x2 > right_region_boundary:
                            right_fit.append((slope, intercept))

            # Funkcija za kreiranje krajnjih tačaka linija
            def make_points(frame, line):
                height, width, _ = frame.shape
                slope, intercept = line
                y1 = height  # bottom of the frame
                y2 = int(y1 * 0.5)  # make points from bottom of the frame up

                # bound the coordinates within the frame
                x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
                x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))

                return [[x1, y1, x2, y2]]

            # Prosečne linije
            if left_fit:
                left_fit_average = np.average(left_fit, axis=0)
                lane_lines.append(make_points(resized_frame, left_fit_average))

            if right_fit:
                right_fit_average = np.average(right_fit, axis=0)
                lane_lines.append(make_points(resized_frame, right_fit_average))

            # Crtanje detektovanih linija
            output_frame = resized_frame.copy()
            for line in lane_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        else:
            output_frame = resized_frame.copy()

        # Postojeća auto mode logika
        control_output = None
        if auto_mode:
            error = 0.0
            if lane_lines:
                centers = []
                for line in lane_lines:
                    for x1, y1, x2, y2 in line:
                        center_line = (x1 + x2) / 2.0
                        centers.append(center_line)
            
                if centers:
                    avg_center = sum(centers) / len(centers)
                    print("avg",avg_center)
                    
                    image_center = width / 2.0
                    print("centar",image_center)
                    # Greška – pozitivna vrednost znači da se sredina linija pomera ulevo od centra slike
                    error = image_center - avg_center

                    # Pretpostavljamo fiksno dt (može se kasnije zameniti merom vremena)
                    dt = 0.01
                    counter=0
                    
                    pid_value = pid_controller.update(error, dt)
                    print("PIDDDDD",pid_value)
                    
                    # Ograničavamo PID izlaz na opseg upravljača (npr. između -25 i 25)
                    steering = max(min(pid_value, 25), -25)
                    # Za brzinu možemo postaviti fiksnu vrednost, npr. 20
                    control_output = {"steer": steering, "speed": 20}

                    # Dodavanje teksta za debagovanje
                    cv2.putText(output_frame, f"Error: {error:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(output_frame, f"PID: {pid_value:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        

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

        return output_frame, control_output
 
    
    def capture_loop(self):
        """Nit koja kontinuirano preuzima frame-ove i ažurira shared buffer."""
        while self._running and self._acquisition_running.is_set():
            try:
                frame = self.camera.capture_array("main", wait=True)
                if frame is not None:
                    self.frame_buffer.update(frame)
                else:
                    self.logger.error("Capture loop: frame je None.")
            except Exception as e:
                self.logger.error(f"Capture loop error: {e}")
            time.sleep(0.001)

    def processing_loop(self):
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

            if not self.frame_buffer.event.wait(timeout=0.005):
                continue
            frame, capture_time = self.frame_buffer.get()
            self.frame_buffer.clear()
            if frame is None:
                continue
            proc_start = time.time()
            processed_frame, control = self.process_frame(frame)
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
                    control = {"steer": 0, "speed": 50}
                    # self.logger.info("Using fixed command: %s", control)
                else:
                    self.logger.info("Computed auto command: %s", control)
                auto_control_message = {"channel": "AutoControl", "data": control}
                self.autoControlSender.send(json.dumps(auto_control_message))
            # Bez dodatnog spavanja

    def run(self):
        """Glavna metoda koja pokreće acquisition i processing niti."""
        self.capture_thread = threading.Thread(target=self.capture_loop, name="CameraCapture")
        self.processing_thread = threading.Thread(target=self.processing_loop, name="CameraProcessing")
        self.capture_thread.start()
        self.processing_thread.start()
        while self._running:
            time.sleep(0.1)
        self._acquisition_running.clear()
        self._processing_running.clear()
        self.capture_thread.join()
        self.processing_thread.join()

    def stop(self):
        """Zaustavlja nit; ukoliko se snima, oslobađa VideoWriter."""
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        super(threadCamera, self).stop()
        if self.yolo_thread is not None:
            self.yolo_thread.stop()
            self.yolo_thread.join()

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

class ObjectDetectionThread(threading.Thread):
    def __init__(self, frame_buffer, queuesList, logger, model_path="Brain_koji_radi/Models/best.pt"):
        super(ObjectDetectionThread, self).__init__()
        self.frame_buffer = frame_buffer
        self.queuesList = queuesList
        self.logger = logger
        self.running = True

        self.model = YOLO(model_path)
        self.cpu_core = 4
        self.set_cpu_affinity()

    def set_cpu_affinity(self):
        pid = os.getpid()
        p = psutil.Process(pid)
        p.cpu_affinity([self.cpu_core])
        p.nice(psutil.HIGH_PRIORITY_CLASS)

    def detect_objects(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append((x1, y1, x2, y2, conf, cls))
        return detections

    def run(self):
        while self.running:
            if not self.frame_buffer.event.wait(timeout=0.01):
                continue
            
            frame, timestamp = self.frame_buffer.get()
            self.frame_buffer.clear()

            if frame is None:
                continue

            start_time = time.time()
            detections = self.detect_objects(frame)
            end_time = time.time()

            processing_time = end_time - start_time
            self.logger.info(f"YOLOv8 processing time: {processing_time:.3f}s, Detections: {len(detections)}")

            for x1, y1, x2, y2, conf, cls in detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, encodedImg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encodedImageData = base64.b64encode(encodedImg).decode("utf-8")
            self.queuesList["yolo_output"].put(encodedImageData)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    # Primer pokretanja kamere u debug režimu
    cam = threadCamera({}, None, True)
    while True:
        if not cam.queuesList["yolo_output"].empty():
            encoded_image = cam.queuesList["yolo_output"].get()
            decoded_image = base64.b64decode(encoded_image)
            np_arr = np.frombuffer(decoded_image, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imshow("YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

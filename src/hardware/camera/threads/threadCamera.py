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

class FrameBuffer:
    """Jednostavni buffer sa kapacitetom 1 koji drži frame i timestamp preuzimanja."""
    def __init__(self):
        self.frame = None
        # Kreiramo poseban queue za YOLO detekcije
#        self.queuesList["yolo_output"] = threading.Queue()

        # Pokrećemo YOLO detekcioni thread
 #       self.yolo_thread = ObjectDetectionThread(self.frame_buffer, self.queuesList, self.logger)
  #      self.yolo_thread.start()

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
    """Thread koji rukuje funkcionalnostima kamere.

    Args:
        queuesList (dict): Rečnik redova gde je ključ tip poruke.
        logger (logging.Logger): Logger za debagovanje.
        debugger (bool): Flag za debagovanje.
    """
    def __init__(self, queuesList, logger, debugger):
        super(threadCamera, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_rate = 7  # FPS
        self.recording = False
        self.video_writer = None

        # Inicijaliziramo executor za paralelnu obradu (ostaje za eventualne dodatne zadatke)
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.recordingSender = messageHandlerSender(self.queuesList, Recording)
        self.mainCameraSender = messageHandlerSender(self.queuesList, mainCamera)
        self.serialCameraSender = messageHandlerSender(self.queuesList, serialCamera)

        self.subscribe()
        self._init_camera()
        self.Queue_Sending()
        self.Configs()

        # Kreiramo zajednički buffer za frame-ove
        self.frame_buffer = FrameBuffer()
        # Flagovi za kontrolu rada acquisition i processing niti
        self._acquisition_running = threading.Event()
        self._acquisition_running.set()
        self._processing_running = threading.Event()
        self._processing_running.set()

    def subscribe(self):
        """Pretplate na poruke (record, brightness, contrast)."""
        self.recordSubscriber = messageHandlerSubscriber(self.queuesList, Record, "lastOnly", True)
        self.brightnessSubscriber = messageHandlerSubscriber(self.queuesList, Brightness, "lastOnly", True)
        self.contrastSubscriber = messageHandlerSubscriber(self.queuesList, Contrast, "lastOnly", True)

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
        Obrada ulaznog frame‑a s fokusom na ROI definisanom trapezom.
        (Ovaj metod ostaje nepromenjen.)
        """
        if frame is None:
            self.logger.error("Primljen main frame je None.")
            return None

        resized_frame = cv2.resize(frame, (1024, 540))
        height, width = resized_frame.shape[:2]
        roi_pts = np.array([
            [0, height],
            [width, height],
            [int(2 * width / 3), int(2 * height / 3)],
            [int(width / 3), int(2 * height / 3)]
        ], np.int32)
        roi_pts[:, 0] = np.clip(roi_pts[:, 0], 0, width - 1)
        roi_pts[:, 1] = np.clip(roi_pts[:, 1], 0, height - 1)
        x, y, w, h = cv2.boundingRect(roi_pts)
        if w == 0 or h == 0:
            self.logger.error("Degenerisana ROI oblast. Preskačem frame.")
            return resized_frame
        roi_img = resized_frame[y:y+h, x:x+w].copy()
        roi_pts_adjusted = roi_pts - [x, y]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts_adjusted], 255)
        self.logger.debug(f"ROI shape: {roi_img.shape}, mask shape: {mask.shape}")
        try:
            roi_only = cv2.bitwise_and(roi_img, roi_img, mask=mask)
        except Exception as e:
            self.logger.error(f"cv2.bitwise_and greška: {e}")
            return resized_frame

        roi_gamma = self.adjust_gamma(roi_only, gamma=1.2)
        gray = cv2.cvtColor(roi_gamma, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (3, 3))
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
        roi_processed = np.copy(roi_img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi_processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            self.logger.debug("Nije pronađena nijedna linija u ROI-ju.")
        roi_final = roi_img.copy()
        roi_final[mask == 255] = roi_processed[mask == 255]
        output_frame = np.copy(resized_frame)
        output_frame[y:y+h, x:x+w] = roi_final
        cv2.polylines(output_frame, [roi_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        return output_frame

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
            # Minimalna pauza da se ne preoptereti kamera
            time.sleep(0.001)

    def processing_loop(self):
        """Nit koja čeka frame u bufferu, obrađuje ga, meri kašnjenje i, ako je snimanje aktivno, zapisuje ga u video."""
        while self._running and self._processing_running.is_set():
            # Prvo proveravamo da li je došla poruka za snimanje
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

            # Čekamo na signal da je novi frame dostupan (timeout 5ms za bržu reakciju)
            if not self.frame_buffer.event.wait(timeout=0.005):
                continue
            frame, capture_time = self.frame_buffer.get()
            self.frame_buffer.clear()
            if frame is None:
                continue
            proc_start = time.time()
            processed_frame = self.process_frame(frame)
            proc_end = time.time()
            capture_delay = proc_start - capture_time
            processing_delay = proc_end - proc_start
            total_delay = proc_end - capture_time
            '''self.logger.info(
                f"Frame delays: capture = {capture_delay:.3f}s, processing = {processing_delay:.3f}s, total = {total_delay:.3f}s"
            )'''
            if processed_frame is None:
                self.logger.error("Obrada frame-a nije uspjela.")
                continue
            if self.recording and self.video_writer is not None:
                self.video_writer.write(processed_frame)
            _, encodedImg = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encodedImageData = base64.b64encode(encodedImg).decode("utf-8")
            self.mainCameraSender.send(encodedImageData)
            self.serialCameraSender.send(encodedImageData)
            # Ne stavljamo dodatni sleep

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
        # Pauza za inicijalizaciju kamere
        time.sleep(2)

class ObjectDetectionThread(threading.Thread):
    def __init__(self, frame_buffer, queuesList, logger, model_path="Brain_koji_radi/Models/best.pt"):
        super(ObjectDetectionThread, self).__init__()
        self.frame_buffer = frame_buffer
        self.queuesList = queuesList
        self.logger = logger
        self.running = True

        # Učitavanje YOLOv8 modela
        self.model = YOLO(model_path)

        # Podesi CPU affinity na core 4
        self.cpu_core = 4  # Možeš menjati ako treba drugi core
        self.set_cpu_affinity()

    def set_cpu_affinity(self):
        """Postavlja da se ovaj thread izvršava isključivo na CPU core 4."""
        pid = os.getpid()
        p = psutil.Process(pid)
        p.cpu_affinity([self.cpu_core])  # Postavlja affinity na core 4
        p.nice(psutil.HIGH_PRIORITY_CLASS)  # Postavlja najviši prioritet

    def detect_objects(self, frame):
        """Pokreće YOLO model i vraća listu detekcija."""
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
        """Glavna petlja za preuzimanje frame-ova iz buffera i detekciju objekata."""
        while self.running:
            # Čekamo na novi frame u bufferu (timeout 10ms)
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

            # Crtanje bounding boxova
            for x1, y1, x2, y2, conf, cls in detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Konvertuj u JPEG format za slanje preko queue-a
            _, encodedImg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encodedImageData = base64.b64encode(encodedImg).decode("utf-8")

            # Pošalji obrađeni frame preko queue sistema
            self.queuesList["yolo_output"].put(encodedImageData)

    def stop(self):
        """Zaustavlja thread."""
        self.running = False



if __name__ == "__main__":
    cam = threadCamera({}, None, True)  # Pokrećemo kameru u debug modu
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

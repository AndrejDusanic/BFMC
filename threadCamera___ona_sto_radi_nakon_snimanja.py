import cv2
import threading
import base64
import picamera2
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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


class threadCamera(ThreadWithStop):
    """Thread koji rukuje funkcionalnostima kamere.

    Args:
        queuesList (dict): Rečnik redova gdje je ključ tip poruke.
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

        # Inicijaliziramo executor za paralelnu obradu frame-ova
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.recordingSender = messageHandlerSender(self.queuesList, Recording)
        self.mainCameraSender = messageHandlerSender(self.queuesList, mainCamera)
        self.serialCameraSender = messageHandlerSender(self.queuesList, serialCamera)

        self.subscribe()
        self._init_camera()
        self.Queue_Sending()
        self.Configs()

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
        Obrada ulaznog frame‑a:
          - Promjena veličine radi optimizacije (1024x540)
          - Gamma korekcija
          - Konverzija u grayscale i blur radi smanjenja šuma
          - Detekcija ivica pomoću Canny metode
          - Primena maske (ROI) na ivice
          - Detekcija linija pomoću Hough transformacije
          - Crtanje detektovanih linija na originalnu sliku
        """
        if frame is None:
            self.logger.error("Primljen main frame je None.")
            return None

        # Promjena veličine
        resized_frame = cv2.resize(frame, (1024, 540))
        # Gamma korekcija
        gamma_corrected = self.adjust_gamma(resized_frame, gamma=1.2)
        # Konverzija u grayscale
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
        # Blur radi smanjenja šuma
        blurred = cv2.blur(gray, (3, 3))
        # Detekcija ivica
        edges = cv2.Canny(blurred, 50, 150)

        # Kreiranje maske – definisanje regiona interesa (ROI)
        mask = np.zeros_like(edges)
        height, width = edges.shape
        polygon = np.array([[(0, height), (width, height), (width, height // 3), (0, height // 3)]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detekcija linija pomoću Hough transformacije
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
        # Kopiramo originalnu (resized) sliku da bismo nacrtali linije
        line_image = np.copy(resized_frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            self.logger.debug("Nije pronađena nijedna linija.")

        return line_image

    def run(self):
        """Glavna petlja niti: obrađuje frame-ove, šalje ih putem gateway‑a te upravlja snimanjem."""
        while self._running:
            try:
                # Provjera poruka za režim snimanja
                recordRecv = self.recordSubscriber.receive()
                if recordRecv is not None:
                    self.recording = bool(recordRecv)
                    if not self.recording and self.video_writer is not None:
                        self.video_writer.release()
                        self.video_writer = None
                    elif self.recording and self.video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        # Snimamo obrađeni frame (1024x540)
                        self.video_writer = cv2.VideoWriter(
                            "output_video_" + str(time.time()) + ".avi",
                            fourcc,
                            self.frame_rate,
                            (1024, 540),
                        )
            except Exception as e:
                self.logger.error(f"Record subscriber error: {e}")

            try:
                # Dohvati glavni frame
                main_frame = self.camera.capture_array("main", wait=True)
                if main_frame is None:
                    self.logger.error("Main frame nije preuzet!")
                    continue
                self.logger.debug(f"Main frame shape: {main_frame.shape}")
                
                # Pokreni obradu glavnog frame‑a u zasebnoj niti
                processed_future = self.executor.submit(self.process_frame, main_frame)

                # Dohvati serial frame
                serial_frame = self.camera.capture_array("lores", wait=True)
                if serial_frame is None:
                    self.logger.error("LoRes frame nije preuzet!")
                    continue
                self.logger.debug(f"LoRes frame type: {type(serial_frame)}")
                if hasattr(serial_frame, 'shape'):
                    self.logger.debug(f"LoRes frame shape: {serial_frame.shape}")
                else:
                    self.logger.debug("LoRes frame nema atribut shape.")

                # Ako je serial_frame tipa bytes, dekodiraj ga
                if isinstance(serial_frame, (bytes, bytearray)):
                    self.logger.debug("LoRes frame je tipa bytes – pokrećem dekodiranje.")
                    nparr = np.frombuffer(serial_frame, np.uint8)
                    serial_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if serial_frame is None:
                        self.logger.error("Dekodiranje lores frame‑a nije uspjelo.")
                        continue
                else:
                    # Ako je raw podatak, pokušaj konvertirati iz YUV u BGR
                    try:
                        serial_frame = cv2.cvtColor(serial_frame, cv2.COLOR_YUV2BGR_I420)
                    except Exception as e:
                        self.logger.error(f"Greška pri cv2.cvtColor: {e}")
                        continue

                # Preuzmi obrađeni glavni frame (sa linijama)
                processed_frame = processed_future.result()
                if processed_frame is None:
                    self.logger.error("Obrada glavnog frame‑a nije uspjela.")
                    continue

                # Ako je snimanje aktivno, snimi obrađeni frame
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(processed_frame)

                # Enkodiraj frame‑ove u JPEG (kvalitet 80)
                _, mainEncodedImg = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                _, serialEncodedImg = cv2.imencode(".jpg", serial_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                mainEncodedImageData = base64.b64encode(mainEncodedImg).decode("utf-8")
                serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8")

                self.mainCameraSender.send(mainEncodedImageData)
                self.serialCameraSender.send(serialEncodedImageData)
            except Exception as e:
                self.logger.error(f"Camera thread error: {e}")

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
        # Pauza za inicijalizaciju kamere
        time.sleep(2)

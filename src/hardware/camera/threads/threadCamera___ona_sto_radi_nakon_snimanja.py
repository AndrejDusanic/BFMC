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
        Obrada ulaznog frame‑a s fokusom na ROI koji je trapez definisan tačkama:
        - Korisničke (donji levi kao početak): (0,0), (width,0), (2/3*width,1/3*height), (1/3*width,1/3*height)
        - Prebačeno u OpenCV koordinatni sistem: (0, height), (width, height), (2/3*width,2/3*height), (1/3*width,2/3*height)
        Izvršava se:
          - Promena veličine slike na 1024x540
          - Izdvajanje ROI‑ja kao trapeza
          - Primena algoritma (gamma, grayscale, blur, Canny, Hough) samo nad ROI‑jem
          - Iscrtavanje detektovanih linija unutar ROI‑ja (zelena)
          - Preklapanje obrađenog ROI‑ja nazad u originalnu sliku
          - Iscrtavanje trapeznog okvira ROI‑ja (crveno)
        """
        if frame is None:
            self.logger.error("Primljen main frame je None.")
            return None

        # Promena veličine
        resized_frame = cv2.resize(frame, (1024, 540))
        height, width = resized_frame.shape[:2]

        ### PROMENA: Definisanje trapeznog ROI-ja prema korisničkim specifikacijama  
        # Korisničke tačke (donji levi kao početak): (0,0), (width,0), (2/3*width,1/3*height), (1/3*width,1/3*height)
        # Prebacujemo u OpenCV koordinatni sistem (gornji levi je početak):
        # p1 = (0, height)
        # p2 = (width, height)
        # p3 = (int(2/3*width), int(2/3*height))
        # p4 = (int(1/3*width), int(2/3*height))
        roi_pts = np.array([
            [0, height],
            [width, height],
            [int(2 * width / 3), int(2 * height / 3)],
            [int(width / 3), int(2 * height / 3)]
        ], np.int32)
        roi_pts = roi_pts.reshape((-1, 1, 2))

        # Dobijamo bounding rectangle trapeznog ROI-ja
        x, y, w, h = cv2.boundingRect(roi_pts)
        roi_img = resized_frame[y:y+h, x:x+w].copy()

        # Podesimo tačke ROI-ja u odnosu na bounding rectangle
        roi_pts_adjusted = roi_pts - [x, y]

        # Kreiramo masku za ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts_adjusted], 255)

        # Izolujemo ROI pomoću maske
        roi_only = cv2.bitwise_and(roi_img, roi_img, mask=mask)

        ### PROMENA: Primena obrade isključivo na ROI
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

        # Kombinujemo obrađeni ROI s originalnim ROI-jem samo unutar maske
        roi_final = roi_img.copy()
        roi_final[mask == 255] = roi_processed[mask == 255]

        # Vraćamo obrađeni ROI u originalnu sliku
        output_frame = np.copy(resized_frame)
        output_frame[y:y+h, x:x+w] = roi_final

        ### PROMENA: Iscrtavamo trapezni ROI okvir (crveno) na izlaznoj slici
        cv2.polylines(output_frame, [roi_pts], isClosed=True, color=(0, 0, 255), thickness=2)

        return output_frame

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
                
                # Obrada glavnog frame-a (samo obrađeni frame se koristi)
                processed_future = self.executor.submit(self.process_frame, main_frame)
                processed_frame = processed_future.result()
                if processed_frame is None:
                    self.logger.error("Obrada glavnog frame-a nije uspjela.")
                    continue

                # Ako je snimanje aktivno, snimi obrađeni frame
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(processed_frame)

                ### PROMENA: Šaljemo samo obrađeni frame (output_frame) na dashboard
                _, encodedImg = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                encodedImageData = base64.b64encode(encodedImg).decode("utf-8")

                self.mainCameraSender.send(encodedImageData)
                self.serialCameraSender.send(encodedImageData)
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

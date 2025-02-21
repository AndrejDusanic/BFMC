# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# To start the project: 
#
#       sudo apt update
#       sudo apt upgrade
#       xargs sudo apt install -y < "requirement.txt" 
#       cd src/dashboard/frontend/
#       curl -fsSL https://fnm.vercel.app/install | bash
#       source ~/.bashrc
#       fnm install --lts
#       npm install -g @angular/cli@17
#       npm install
#       if needed: npm audit fix
#
# ===================================== GENERAL IMPORTS ==================================
#!/usr/bin/env python3

#!/usr/bin/env python3
"""
Refaktorisana verzija main.py sa poboljšanim upravljanjem procesima,
postavljanjem CPU affinity-ja i prioritetom:
- Thread kamere se izvršava isključivo na core 1 (CPU ID 1).
- Thread upravljanja Nucleo-om se izvršava isključivo na core 4 (CPU ID 3)
  uz postavljanje najvišeg prioriteta (-20).
- Thread dashboard-a se izvršava isključivo na core 2 (CPU ID 2).
Ostali procesi se izvršavaju bez pinovanja.
"""

import sys
import subprocess
import time
import os
from multiprocessing import Process, Queue, Event
import logging
import json

# Dodajemo putanju (ukoliko je potrebno)
sys.path.append(".")

# ===================================== GENERAL SETUP ==================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

queueList = {
    "Critical": Queue(),
    "Warning": Queue(),
    "General": Queue(),
    "Config": Queue(),
}

# ===================================== PROCESS IMPORTS ==================================
from src.gateway.processGateway import processGateway
from src.dashboard.processDashboard import processDashboard
from src.hardware.camera.processCamera import processCamera
from src.hardware.serialhandler.processSerialHandler import processSerialHandler
from src.data.Semaphores.Semaphores import processSemaphores
from src.data.TrafficCommunication.processTrafficCommunication import processTrafficCommunication
from src.utils.ipManager.IpReplacement import IPManager

# ===================================== CONFIG FLAGS ==================================
ENABLE_DASHBOARD = True
ENABLE_CAMERA = True
ENABLE_SEMAPHORES = False
ENABLE_TRAFFIC_COMMUNICATION = False
ENABLE_SERIAL_HANDLER = True

# ===================================== HELPER FUNKCIJE ==================================
def set_process_priority():
    """
    Primer za postavljanje prioriteta trenutnog procesa.
    Ovaj primer koristi os.nice; može se proširiti pomoću os.setpriority.
    """
    try:
        current_nice = os.nice(0)
        os.nice(-5)
        logger.info(f"Process priority changed from {current_nice} to {os.nice(0)}")
    except Exception as e:
        logger.warning(f"Unable to change process priority: {e}")

def start_processes(processes):
    """
    Pokreće sve procese u listi, a zatim za specifične procese postavlja CPU affinity i,
    za Nucleo, najviši prioritet.
    - Kamera: affinity postavljen na core 1 (CPU ID 1)
    - Dashboard: affinity postavljen na core 2 (CPU ID 2)
    - Nucleo: affinity postavljen na core 3 (CPU ID 3) i prioritet -20
    """
    for proc in processes:
        proc.start()
        logger.info(f"Process started: {proc}")
    # Kratka pauza kako bi se procesi pokrenuli i dobili PID
    time.sleep(1)
    # Postavljanje CPU affinity i prioriteta za određene procese
    for proc in processes:
        if isinstance(proc, processCamera):
            try:
                os.sched_setaffinity(proc.pid, {1})
                os.setpriority(os.PRIO_PROCESS, proc.pid, 0)
                logger.info(f"Set CPU affinity of camera process {proc.pid} to core 1")
            except Exception as e:
                logger.warning(f"Could not set affinity/priority for CAMERA process: {e}")

        elif isinstance(proc, processSerialHandler):
            try:
                os.sched_setaffinity(proc.pid, {2})
                logger.info(f"Set CPU affinity of Nucleo process {proc.pid} to core 2")
                os.setpriority(os.PRIO_PROCESS, proc.pid, 0)
            except Exception as e:
                logger.warning(f"Could not set affinity/priority for Nucleo process: {e}")
        elif isinstance(proc, processDashboard):
            try:
               #os.sched_setaffinity(proc.pid, {3})
                logger.info(f"Dasboart init started")
            except Exception as e:
                logger.warning(f"Could not set affinity for dashboard process: {e}")

def initialize_processes(queueList, logger):
    """
    Inicijalizuje i vraća listu svih procesa koji treba da se pokrenu.
    """
    processes = []

    # Inicijalizacija gateway procesa – pokreće se odmah
    gateway = processGateway(queueList, logger)
    gateway.start()
    logger.info("Gateway process started.")

    # Izmena IP adrese u frontend datoteci
    ip_file_path = './src/dashboard/frontend/src/app/webSocket/web-socket.service.ts'
    ip_manager = IPManager(ip_file_path)
    ip_manager.replace_ip_in_file()
    logger.info("IP address updated in frontend file.")

    # Inicijalizacija ostalih komponenti
    if ENABLE_DASHBOARD:
        dashboard_proc = processDashboard(queueList, logger, debugging=False)
        processes.append(dashboard_proc)

    if ENABLE_CAMERA:
        camera_proc = processCamera(queueList, logger, debugging=False)
        processes.append(camera_proc)

    if ENABLE_SEMAPHORES:
        semaphores_proc = processSemaphores(queueList, logger, debugging=False)
        processes.append(semaphores_proc)

    if ENABLE_TRAFFIC_COMMUNICATION:
        traffic_proc = processTrafficCommunication(queueList, logger, 3, debugging=False)
        processes.append(traffic_proc)

    if ENABLE_SERIAL_HANDLER:
        serial_proc = processSerialHandler(queueList, logger, debugging=False)
        processes.append(serial_proc)

    return processes, gateway

def stop_processes(processes, gateway):
    """
    Zaustavlja sve pokrenute procese i gateway.
    """
    logger.info("Shutdown signal received. Stopping processes...")
    for proc in reversed(processes):
        try:
            proc.stop()  # Pretpostavljamo da svaka klasa implementira metodu stop()
            proc.join(timeout=5)
            logger.info(f"Process stopped: {proc}")
        except Exception as e:
            logger.error(f"Error stopping process {proc}: {e}")
    try:
        gateway.stop()
        logger.info(f"Gateway process stopped: {gateway}")
    except Exception as e:
        logger.error(f"Error stopping gateway process: {e}")

def print_banner():
    c4_bomb = r"""
     ______________________
    /                      \
   | [██████]     [██████] |
   | [██████]     [██████] |
   | [██████]     [██████] |
   |   TIMER:       00:10  |
    \______________________/
       LET'S GO!!!
    Press ctrl+C to close
    """
    print(c4_bomb)

# ===================================== MAIN EXECUTION ==================================
def main():
    # Opcionalno – postavljanje prioriteta za glavni proces
    set_process_priority()

    processes, gateway = initialize_processes(queueList, logger)
    start_processes(processes)

    # Kratka pauza da se svi procesi inicijalizuju
    time.sleep(10)
    print_banner()

    # Glavna petlja – čeka prekid (ctrl+C)
    blocker = Event()
    try:
        blocker.wait()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping all processes...\n")
        stop_processes(processes, gateway)
        print("\nAll processes stopped. Exiting program.")
        sys.exit(0)

if __name__ == '__main__':
    main()

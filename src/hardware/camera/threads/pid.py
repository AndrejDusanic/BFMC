# src/hardware/camera/threads/pid.py

from flask import Flask
from flask_socketio import SocketIO
import time

#-------------------------------
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0

    def update(self, error, dt):
        """Izračunava PID izlaz na osnovu trenutne greške i proteklog vremena dt."""
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output

# Globalni objekat PID kontrolera
pid_controller = PIDController()

# Globalna zastavica da li je auto režim aktivan
auto_mode = True

def set_auto_mode(mode: bool):
    """Postavlja globalni auto_mode – kada je True, PID se koristi za automatsko upravljanje."""
    global auto_mode
    auto_mode = mode
#------------------------------
app = Flask(__name__)

app.config['SECRET_KEY'] = 'tajni_kljuc'

@app.route('/')
def index():
    return "Server je pokrenut!"
 
# Dozvoljavamo CORS kako bi Angular aplikacija sa druge adrese mogla da se poveže
socketio = SocketIO(app, cors_allowed_origins="*")

speed = 0
steer = 0

def send_periodic_updates():
    global speed, steer
    while True:
        speed = speed + 1
        steer = steer + 1

        if speed>30:
            speed=0
        if steer>20:
            steer=0

        socketio.emit('message_about_speed', {"speed": str(speed)})
        socketio.emit('message_about_steering_angle', {"steer": str(speed)})
        socketio.sleep(1)  # Adjust the interval as needed (e.g., every 1 second)


# Kada se klijent poveže, obaveštavamo ga (i server)
@socketio.on('connect')
def handle_connect():
    print("Klijent se povezao!")
    # Emitujemo događaj 'after connect' kao što Angular očekuje
    #socketio.emit('after connect', {"message": "Povezano!"})
    socketio.emit('message_about_speed', {"speed": "0"})
    socketio.emit('message_about_steering_angle', {"steer": "0"})

   # Start background task (only once, when the first client connects)
    socketio.start_background_task(send_periodic_updates)

#@socketio.on('second')
#def handle_second(data):
#    socketio.emit('message_about_speed', {"speed": "-20"})
#    socketio.emit('message_about_steering_angle', {"steer": "-20"})

#    time.sleep(50)    
#    socketio.emit('message_about_speed', {"speed": "-20"})
#    socketio.emit('message_about_steering_angle', {"steer": "-20"})


# Ova funkcija prima poruke sa kanala 'message'
@socketio.on('message')
def handle_message(data):
    print(data)
    # Ovde možete obraditi primljenu poruku, npr. postaviti AutoMode ili izvršiti drugu logiku.
    # Kao odgovor možemo poslati poruku nazad
    #socketio.send({"status": "poruka primljena"})

@socketio.on("debbuging_message")
def handle_debbuging_message(data):
    print(data)

if __name__ == '__main__':
    # Pokrećemo server na portu 5001
    socketio.run(app, host='172.20.10.3', port=5001, debug=True)

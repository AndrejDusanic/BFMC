# src/hardware/camera/threads/pid.py

from flask import Flask
from flask_socketio import SocketIO
import time
#-------------------------------
import json
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
        #speed = shared_data.speed
        #print("shared speed",shared_data.speed)
        print("speed",speed)
        #steer = shared_data.steer
        #print("shared steer",shared_data.steer)
        print("steer",steer)


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

@socketio.on('control_output')
def handle_control_output(data):
    global speed, steer
    if isinstance(data, dict):
        steer=data.get("steer",steer)
        speed=data.get("speed",speed)
    else:
        try:
            d = json.loads(data)
            steer = d.get("steer", steer)
            speed = d.get("speed", speed)
        except Exception as e:
            print("Greška pri parsiranju control_output:", e)
    print("Primljen control_output:", {"steer": steer, "speed": speed})

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

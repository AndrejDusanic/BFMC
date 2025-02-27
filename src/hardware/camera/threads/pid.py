# src/hardware/camera/threads/pid.py


from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tajni_kljuc'

@app.route('/')
def index():
    return "Server je pokrenut!"
 
# Dozvoljavamo CORS kako bi Angular aplikacija sa druge adrese mogla da se poveže
socketio = SocketIO(app, cors_allowed_origins="*")

# Kada se klijent poveže, obaveštavamo ga (i server)
@socketio.on('connect')
def handle_connect():
    print("Klijent se povezao!")
    # Emitujemo događaj 'after connect' kao što Angular očekuje
    socketio.emit('after connect', {"message": "Povezano!"})

# Ova funkcija prima poruke sa kanala 'message'
@socketio.on('message')
def handle_message(data):
    print("Primljena poruka:", data)
    # Ovde možete obraditi primljenu poruku, npr. postaviti AutoMode ili izvršiti drugu logiku.
    # Kao odgovor možemo poslati poruku nazad
    socketio.send({"status": "poruka primljena"})

if __name__ == '__main__':
    # Pokrećemo server na portu 5005, što odgovara URL-u u Angular aplikaciji
    socketio.run(app, host='172.20.10.3', port=5001, debug=True)

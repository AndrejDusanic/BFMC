
// src/app/cluster/state-switch/state-switch.component.ts

// Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
// All rights reserved.

import { Component, HostListener, OnInit } from '@angular/core';
import { WebSocketService } from '../../webSocket/web-socket.service';
import { NgFor, NgIf } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { ClusterService } from '../cluster.service';

//====ODAVDE PA ISPOD==========

let car_speed: number = 0; 
let car_steering_angle: number = 0;


import { io, Socket } from 'socket.io-client';

// Connect to your Flask server (make sure the URL and port match your server settings)
const socket: Socket = io('http://172.20.10.3:5001');

socket.on('connect', () => {
  // console.log('Connected to the Flask server');
  // Optionally send a message after connecting
  //socket.emit('message', { message: 'Hello from TypeScript client!' });
});

socket.on('message_about_speed', (data: { speed: string }) => {
  //console.log('Message from server:', data);
  //socket.emit('message', {message: 'Primio sam status sa servera da je primio moju prvu poruku!'});
  car_speed = Number(data.speed);
  socket.emit('message', {message: car_speed})
 // setTimeout(() => {
  //  socket.emit('second', { second: 'Salji drugu komandu!' });
   // }, 40000); // 40 seconds = 40000 milliseconds

});

socket.on('message_about_steering_angle', (data: { steer: string }) => {
  //console.log('Message from server:', data);
  //socket.emit('message', {message: 'Primio sam status sa servera da je primio moju prvu poruku!'});
  car_steering_angle = Number(data.steer);
  socket.emit('message', {message: car_steering_angle})
});

//====DO OVDE=====

@Component({
  selector: 'app-state-switch',
  standalone: true,
  imports: [NgFor, NgIf, MatIconModule],
  templateUrl: './state-switch.component.html',
  styleUrls: ['./state-switch.component.css']
})
export class StateSwitchComponent implements OnInit {
  public states: string[] = ['stop', 'manual', 'legacy', 'auto'];
  public currentStateIndex: number = 0;

  public isMobile: boolean = false;

  // Promenljive za prikaz auto kontrolnih vrednosti
  public autoSteer: number = 0;
  public autoSpeed: number = 0;

  private activeKey: string | null = null;

  private speed: number = 0;
  private speedIncrement: number = 5;
  private maxSpeed: number = 50;
  private minSpeed: number = -50;

  private steer: number = 0;
  private lastSteer: number = 0;
  private steerIncrement: number = 5;
  private steerDecrement: number = 5;
  private steerInterval: any;
  private steerDecreaseInterval: any;
  private isSteering: boolean = false;
  private maxSteer: number = 25;
  private minSteer: number = -25;

 
  constructor(private webSocketService: WebSocketService, 
              private clusterService: ClusterService) { }

  ngOnInit() {
    this.clusterService.isMobileDriving$.subscribe(isMobileDriving => {
      this.isMobile = isMobileDriving;
    });
    // Pretplata na sve neobrađene događaje sa servera i filtriranje AutoControl poruka
    this.webSocketService.receiveUnhandledEvents().subscribe(event => {
      if (event.channel === "AutoControl") {
        try {
          const control = JSON.parse(event.data);
          this.autoSteer = control.steer;
          this.autoSpeed = control.speed;
          // Poziv metode applyAutoControl – sada implementiramo ovu metodu
          this.applyAutoControl(control.steer, control.speed);
        } catch (err) {
          console.error("Greška pri parsiranju AutoControl poruke:", err);
        }
      }
    });
  }

    //this.steer = isNaN(car_steering_angle) ? 0 : car_steering_angle;
  // Dodata metoda applyAutoControl koja obrađuje primljene auto komande.
  // Ovde možete implementirati potrebnu logiku – trenutno samo logujemo vrednosti.
  applyAutoControl(steer: number, speed: number): void {
    console.log(`Primljena auto komanda - Steer: ${steer}, Speed: ${speed}`);
    // Možete dodati dodatnu logiku za upravljanje, ažuriranje UI ili slanje naredbi
  }

  @HostListener('window:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (this.currentState === 'manual') {
      if (this.activeKey === event.key) return;
      this.activeKey = event.key;
      switch(event.key) {
        case 'w':
          this.increaseSpeed();
          //this.activateAutoControl();

          //if (!this.isSteering) {
          //  this.isSteering = true;
          //  this.stopDecreasingSteering();
          //  this.startSteeringRight();
          //}

          break;
        case 's':
          this.decreaseSpeed();
          //this.deactivateAutoControl();
          break;
        case 'a':
          if (!this.isSteering) { 
            this.isSteering = true;
            this.stopDecreasingSteering();
            this.startSteeringLeft();
          }
          break;
        case 'd':
          if (!this.isSteering) { 
            this.isSteering = true;
            this.stopDecreasingSteering();
            this.startSteeringRight();  
          }
          break;
        default:
          break;
      }
    }
  }

  @HostListener('window:keyup', ['$event'])
  handleKeyUp(event: KeyboardEvent) {
    if (this.currentState === 'manual' && this.activeKey === event.key) {
      this.activeKey = null;
      this.stopSteering();
      this.startDecreasingSteer();
    }
  }


activateAutoControl(): void {
    this.speed=0
    this.steer=0

    this.speed = isNaN(car_speed) ? 0 : car_speed;
    this.steer = isNaN(car_steering_angle) ? 0 : car_steering_angle;

  // Send SteerMotor command immediately
  this.webSocketService.sendMessageToFlask(
    `{"Name": "SteerMotor", "Value": "${this.steer * 10}"}`
  );

  // Wait 50ms before sending SpeedMotor command
  setTimeout(() => {
    this.webSocketService.sendMessageToFlask(
      `{"Name": "SpeedMotor", "Value": "${this.speed * 10}"}`
    );

    // Wait another 50ms before stopping (nothing happens after this)
    setTimeout(() => {
      console.log("Completed Steer and Speed commands.");
    }, 50);

  }, 50);
    // socket.emit('debbuging_message', {debbuging_message: 'Evo me izvrsio se activateAutoControl!'})
}

deactivateAutoControl(): void {
  this.speed=0
  this.steer=0

  this.webSocketService.sendMessageToFlask(`{"Name": "SpeedMotor", "Value": "${this.speed*10}"}`);
  this.webSocketService.sendMessageToFlask(`{"Name": "SteerMotor", "Value": "${this.steer*10}"}`);
}


  setState(index: number) {
    if (this.currentState === 'manual' && this.currentState !== this.states[index]) {
      this.speedReset();
      this.steerReset();
    }

    this.currentStateIndex = index;    
    this.clusterService.updateDrivingMode(this.states[index]);
    // Slanje komande za DrivingMode
    this.webSocketService.sendMessageToFlask(`{"Name": "DrivingMode", "Value": "${this.states[index]}"}`);
    // Dodatno, ako je stanje "auto", šaljemo i AutoMode komandu
    if (this.states[index] === 'auto') {
      this.webSocketService.sendMessageToFlask(`{"Name": "AutoMode", "Value": "true"}`);
      setInterval(() => {
        this.activateAutoControl();
      }, 100); // Runs every 300ms

      //this.activateAutoControl();
     // setInterval(this.activateAutoControl,1000);
    } else {
      this.webSocketService.sendMessageToFlask(`{"Name": "AutoMode", "Value": "false"}`);
      this.deactivateAutoControl();
    }
  }

  get currentState() {
    return this.states[this.currentStateIndex];
  }

  getSliderPosition(index: number): string {
    const totalStates = this.states.length;
    const percentage = (index / totalStates) * 100;
    return `calc(${percentage}%)`;
  }

  public onButtonPress(direction: string): void { 
    if (direction === "left") {
      this.stopDecreasingSteering();
      this.startSteeringLeft();
    } else if (direction === "right") {
      this.stopDecreasingSteering();
      this.startSteeringRight();
    }
  }

  public onButtonRelease(): void { 
    this.stopSteering();
    this.startDecreasingSteer();
  }

  public increaseSpeed(): void {
    this.speed += this.speedIncrement;
    if (this.speed > this.maxSpeed) {
      this.speed = this.maxSpeed;
    }
    this.webSocketService.sendMessageToFlask(`{"Name": "SpeedMotor", "Value": "${this.speed * 10}"}`);
  }

  public decreaseSpeed(): void {
    this.speed -= this.speedIncrement;
    if (this.speed < this.minSpeed) {
      this.speed = this.minSpeed;
    }
    this.webSocketService.sendMessageToFlask(`{"Name": "SpeedMotor", "Value": "${this.speed * 10}"}`);
  }

  private startSteeringRight() {
    this.steerInterval = setInterval(() => {
      this.steer += this.steerIncrement;

      //this.steer = isNaN(car_steering_angle) ? 0 : car_steering_angle;

      if (this.steer > this.maxSteer) {
        this.steer = this.maxSteer;
      }
      if (this.lastSteer !== this.maxSteer) { 
        this.webSocketService.sendMessageToFlask(`{"Name": "SteerMotor", "Value": "${this.steer * 10}"}`);
      }
      this.lastSteer = this.steer;
    }, 50);
  }
    
  private startSteeringLeft() {
    this.steerInterval = setInterval(() => {
      this.steer -= this.steerIncrement;

      //this.steer = isNaN(car_steering_angle) ? 0 : -1*car_steering_angle;

      if (this.steer < this.minSteer) {
        this.steer = this.minSteer;
      }
      if (this.lastSteer !== this.minSteer) { 
        this.webSocketService.sendMessageToFlask(`{"Name": "SteerMotor", "Value": "${this.steer * 10}"}`);
      }
      this.lastSteer = this.steer;
    }, 50);
  }

  private startDecreasingSteer() { 
    setInterval(() => {
      if (this.steer === 0 || this.isSteering) return;
      if (this.steer < 0) {
        this.steer += this.steerDecrement;
        if (this.steer > 0) {
          this.steer = 0;
        }
      }
      if (this.steer > 0) {
        this.steer -= this.steerDecrement;
        if (this.steer < 0) {
          this.steer = 0;
        }
      }
      this.webSocketService.sendMessageToFlask(`{"Name": "SteerMotor", "Value": "${this.steer * 10}"}`);
    }, 100);
  }

  private speedReset(): void { 
    this.speed = 0;
    this.webSocketService.sendMessageToFlask(`{"Name": "SpeedMotor", "Value": "${this.speed * 10}"}`);
  }

  private steerReset(): void { 
    this.steer = 0;
    this.webSocketService.sendMessageToFlask(`{"Name": "SteerMotor", "Value": "${this.steer * 10}"}`);
  }

  private stopSteering() {
    if (this.steerInterval) {
      clearInterval(this.steerInterval); 
      this.steerInterval = null; 
    }
    this.isSteering = false; 
  }

  private stopDecreasingSteering() {
    // Implementacija za zaustavljanje intervala, ako je potrebno
if (this.steerDecreaseInterval) { 
      clearInterval(this.steerDecreaseInterval);
      this.steerDecreaseInterval = null;
    }  
}

  getSliderWidth(): string {
    return `calc(100% / ${this.states.length})`;
  }

  getSliderColor() {
    if (this.currentState === 'legacy') {
      return '#2b8fd1';
    }
    if (this.currentState === 'manual') {
      return '#f0ad4e';
    }
    if (this.currentState === 'stop') {
      return '#d9534f';
    }
    if (this.currentState === 'auto') {
      return '#5cb85c';
    }
    return '#2b8fd1';
  }
}

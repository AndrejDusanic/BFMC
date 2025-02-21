// Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this
//     list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import { Injectable } from '@angular/core';
import { Socket } from 'ngx-socket-io';
import { Observable, Subject } from 'rxjs';
import { throttleTime } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class WebSocketService {
  private webSocket: Socket;
  private eventSubject = new Subject<{ channel: string, data: any }>();
  private handledEvents = new Set([
    'memory_channel',
    'cpu_channel',
    'disk_channel',
    'webCamera',
    'Location',
    'Cars',
    'Semaphores',
    'after connect',
    'InstantConsumption',
    'loadBack',
    'WarningSignal',
    'response',
    'BatteryLvl',
    'ResourceMonitor',
    'serialCamera',
    'Recording',
    'CurrentSpeed',
    'CurrentSteer',
    'EnableButton'
  ]);
  
  constructor() {
    // URL postavljen je na lokalnu mrežu. Ako se RPi i klijent nalaze na istoj mreži,
    // to može dodatno smanjiti latenciju.
    this.webSocket = new Socket({
      url: "http://172.20.10.3:5005",
      options: {},
    });

    // Slušamo sve događaje sa WebSocket servera.
    this.webSocket.onAny((eventName: string, data: any) => {
      if (!this.handledEvents.has(eventName)) {
        this.eventSubject.next({ channel: eventName, data });
      }
    });
  }

  // Metoda za slanje poruka (npr. handshake) na Flask server
  sendMessageToFlask(message: any) {
    this.webSocket.emit('message', message);
  }

  SaveTable(message: any) {
    this.webSocket.emit('save', message);
  }

  LoadTable(message: any) {
    this.webSocket.emit('load', message);
  }

  receiveSessionAccess(): Observable<any> {
    return this.webSocket.fromEvent('session_access');
  }

  // Primena throttleTime operatora da se uzima samo poslednja poruka u svakom intervalu od 33ms (~30 fps)
  receiveMemoryUsage(): Observable<any> {
    return this.webSocket.fromEvent('memory_channel').pipe(throttleTime(33));
  }

  receiveImuData(): Observable<any> {
    return this.webSocket.fromEvent('ImuData');
  }

  receiveResourceMonitor(): Observable<any> {
    return this.webSocket.fromEvent('ResourceMonitor');
  }

  receiveWarningSignal(): Observable<any> {
    return this.webSocket.fromEvent('WarningSignal');
  }

  receiveLoadTable(): Observable<any> {
    return this.webSocket.fromEvent('loadBack');
  }

  receiveCpuUsage(): Observable<any> {
    return this.webSocket.fromEvent('cpu_channel');
  }

  // Na kanalu za kamere koristimo throttleTime kako bismo izbacili stare frame-ove i uzeli samo najnoviji u svakom intervalu
  receiveCamera(): Observable<any> {
    return this.webSocket.fromEvent('serialCamera').pipe(
      throttleTime(33, undefined, { leading: false, trailing: true })
    );
  }

  receiveLocation(): Observable<any> {
    return this.webSocket.fromEvent('Location');
  }

  receiveEnableButton(): Observable<any> {
    return this.webSocket.fromEvent('EnableButton');
  }

  receiveCars(): Observable<any> {
    return this.webSocket.fromEvent('Cars');
  }

  receiveInstantConsumption(): Observable<any> {
    return this.webSocket.fromEvent('InstantConsumption');
  }

  receiveBatteryLevel(): Observable<any> {
    return this.webSocket.fromEvent('BatteryLvl');
  }

  receiveSemaphores(): Observable<any> {
    return this.webSocket.fromEvent('Semaphores');
  }

  receiveCurrentSpeed(): Observable<any> {
    return this.webSocket.fromEvent('CurrentSpeed');
  }

  receiveCurrentSteer(): Observable<any> {
    return this.webSocket.fromEvent('CurrentSteer');
  }

  onConnect(): Observable<any> {
    console.log("connected");
    return this.webSocket.fromEvent('after connect');
  }

  disconnectSocket() {
    this.webSocket.disconnect();
  }

  receiveUnhandledEvents(): Observable<{ channel: string, data: any }> {
    return this.eventSubject.asObservable();
  }
}

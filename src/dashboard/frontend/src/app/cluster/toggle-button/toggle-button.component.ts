import { WebSocketService } from './../../webSocket/web-socket.service';
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common'

@Component({
  selector: 'app-toggle-button',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './toggle-button.component.html',
  styleUrls: ['./toggle-button.component.css']
})
export class ToggleButtonComponent {
    output: boolean = false;
    text: string = "line detection"
  
    constructor( private webSocketService: WebSocketService) { }
  
    changeState() {
      if (this.output == false) {
        this.output = true;
        this.text = "sign detection"
      }
      else {
        this.output = false;
        this.text = "line detection"
      }
  
      this.webSocketService.sendMessageToFlask(`{"Output:" ${this.output}"}`);
    }
  
    getButtonColor() {
      if (this.output === true) { 
        return "#5cb85c";
      }
  
      return "#d9534f";
    }
}

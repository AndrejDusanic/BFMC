import {
  esm_exports,
  lookup
} from "./chunk-KF4FKHI7.js";
import {
  InjectionToken,
  NgModule,
  setClassMetadata,
  ɵɵdefineInjector,
  ɵɵdefineNgModule
} from "./chunk-75RU2XC4.js";
import "./chunk-4N4GOYJH.js";
import "./chunk-5OPE3T2R.js";
import {
  Observable,
  share
} from "./chunk-FHTVLBLO.js";
import "./chunk-7GOANPIK.js";

// node_modules/ngx-socket-io/fesm2022/ngx-socket-io.mjs
var WrappedSocket = class {
  config;
  subscribersCounter = {};
  eventObservables$ = {};
  ioSocket;
  emptyConfig = {
    url: "",
    options: {}
  };
  constructor(config) {
    this.config = config;
    if (config === void 0) {
      config = this.emptyConfig;
    }
    const url = config.url;
    const options = config.options;
    const ioFunc = lookup ? lookup : esm_exports;
    this.ioSocket = ioFunc(url, options);
  }
  of(namespace) {
    this.ioSocket.of(namespace);
  }
  on(eventName, callback) {
    this.ioSocket.on(eventName, callback);
  }
  once(eventName, callback) {
    this.ioSocket.once(eventName, callback);
  }
  connect(callback) {
    return this.ioSocket.connect(callback);
  }
  disconnect(_close) {
    return this.ioSocket.disconnect.apply(this.ioSocket, arguments);
  }
  emit(_eventName, ..._args) {
    return this.ioSocket.emit.apply(this.ioSocket, arguments);
  }
  removeListener(_eventName, _callback) {
    return this.ioSocket.removeListener.apply(this.ioSocket, arguments);
  }
  removeAllListeners(_eventName) {
    return this.ioSocket.removeAllListeners.apply(this.ioSocket, arguments);
  }
  fromEvent(eventName) {
    if (!this.subscribersCounter[eventName]) {
      this.subscribersCounter[eventName] = 0;
    }
    this.subscribersCounter[eventName]++;
    if (!this.eventObservables$[eventName]) {
      this.eventObservables$[eventName] = new Observable((observer) => {
        const listener = (data) => {
          observer.next(data);
        };
        this.ioSocket.on(eventName, listener);
        return () => {
          this.subscribersCounter[eventName]--;
          if (this.subscribersCounter[eventName] === 0) {
            this.ioSocket.removeListener(eventName, listener);
            delete this.eventObservables$[eventName];
          }
        };
      }).pipe(share());
    }
    return this.eventObservables$[eventName];
  }
  fromOneTimeEvent(eventName) {
    return new Promise((resolve) => this.once(eventName, resolve));
  }
  listeners(eventName) {
    return this.ioSocket.listeners(eventName);
  }
  listenersAny() {
    return this.ioSocket.listenersAny();
  }
  listenersAnyOutgoing() {
    return this.ioSocket.listenersAnyOutgoing();
  }
  off(eventName, listener) {
    if (!eventName) {
      return this.ioSocket.offAny();
    }
    if (eventName && !listener) {
      return this.ioSocket.off(eventName);
    }
    return this.ioSocket.off(eventName, listener);
  }
  onAny(callback) {
    return this.ioSocket.onAny(callback);
  }
  onAnyOutgoing(callback) {
    return this.ioSocket.onAnyOutgoing(callback);
  }
  prependAny(callback) {
    return this.ioSocket.prependAny(callback);
  }
  prependAnyOutgoing(callback) {
    return this.ioSocket.prependAnyOutgoing(callback);
  }
  timeout(value) {
    return this.ioSocket.timeout(value);
  }
  volatile() {
    return this.ioSocket.volatile;
  }
};
function SocketFactory(config) {
  return new WrappedSocket(config);
}
var SOCKET_CONFIG_TOKEN = new InjectionToken("__SOCKET_IO_CONFIG__");
var SocketIoModule = class _SocketIoModule {
  static forRoot(config) {
    return {
      ngModule: _SocketIoModule,
      providers: [{
        provide: SOCKET_CONFIG_TOKEN,
        useValue: config
      }, {
        provide: WrappedSocket,
        useFactory: SocketFactory,
        deps: [SOCKET_CONFIG_TOKEN]
      }]
    };
  }
  static ɵfac = function SocketIoModule_Factory(__ngFactoryType__) {
    return new (__ngFactoryType__ || _SocketIoModule)();
  };
  static ɵmod = ɵɵdefineNgModule({
    type: _SocketIoModule
  });
  static ɵinj = ɵɵdefineInjector({});
};
(() => {
  (typeof ngDevMode === "undefined" || ngDevMode) && setClassMetadata(SocketIoModule, [{
    type: NgModule,
    args: [{}]
  }], null, null);
})();
export {
  WrappedSocket as Socket,
  SocketIoModule
};
//# sourceMappingURL=ngx-socket-io.js.map

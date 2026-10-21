// Custom HOCKEY-V2 receiver. ESP32 Arduino 2.x BLE API, PlatformIO espressif32.
// pins.h must define LEFT_IN1, LEFT_IN2, RIGHT_IN1, RIGHT_IN2 for the actual
// H-bridge. Verify its truth table: this example assumes dual-PWM IN1/IN2.
#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include "pins.h"
#include "protocol.h"

constexpr char SERVICE[]="6e400001-b5a3-f393-e0a9-e50e24dcca9e";
constexpr char RX_UUID[]="6e400002-b5a3-f393-e0a9-e50e24dcca9e";
constexpr char TX_UUID[]="6e400003-b5a3-f393-e0a9-e50e24dcca9e";
portMUX_TYPE mux=portMUX_INITIALIZER_UNLOCKED;
int16_t requestedLeft=0,requestedRight=0;
uint32_t expires=0;
uint16_t lastSeq=0;
bool haveSeq=false;

void stopRequest(){portENTER_CRITICAL(&mux);requestedLeft=requestedRight=0;expires=0;haveSeq=false;portEXIT_CRITICAL(&mux);}
class Receiver:public BLECharacteristicCallbacks{
  void onWrite(BLECharacteristic* c) override{
    std::string v=c->getValue();if(v.size()!=13)return;
    hockey::Command cmd;
    if(!hockey::decode((const uint8_t*)v.data(),v.size(),cmd))return;
    uint16_t seq=cmd.sequence,ttl=cmd.ttl;int16_t l=cmd.left,r=cmd.right;
    portENTER_CRITICAL(&mux);
    // uint16 wraparound-safe ordering; STOP is always accepted.
    if(!haveSeq||hockey::newer(seq,lastSeq)||(l==0&&r==0)){
      requestedLeft=l;requestedRight=r;expires=millis()+ttl;
      if(!haveSeq||hockey::newer(seq,lastSeq))lastSeq=seq;
      haveSeq=true;
    }
    portEXIT_CRITICAL(&mux);
  }
};
class Connections:public BLEServerCallbacks{
  void onDisconnect(BLEServer* server) override{stopRequest();server->startAdvertising();}
};
void wheel(int pwm,int a,int b){ledcWrite(a,pwm>0?pwm:0);ledcWrite(b,pwm<0?-pwm:0);}
void setup(){
  Serial.begin(115200);
  const int pins[]={LEFT_IN1,LEFT_IN2,RIGHT_IN1,RIGHT_IN2};
  for(int i=0;i<4;i++){ledcSetup(i,20000,8);ledcAttachPin(pins[i],i);ledcWrite(i,0);}
  BLEDevice::init("Hockey-V2");BLEServer* server=BLEDevice::createServer();server->setCallbacks(new Connections());
  BLEService* service=server->createService(SERVICE);
  auto tx=service->createCharacteristic(TX_UUID,BLECharacteristic::PROPERTY_READ);tx->setValue("HOCKEY-V2");
  auto rx=service->createCharacteristic(RX_UUID,BLECharacteristic::PROPERTY_WRITE);rx->setCallbacks(new Receiver());
  service->start();server->getAdvertising()->addServiceUUID(SERVICE);server->startAdvertising();
}
void loop(){
  int l,r;portENTER_CRITICAL(&mux);
  if(expires==0||int32_t(millis()-expires)>=0)requestedLeft=requestedRight=0;
  l=requestedLeft;r=requestedRight;portEXIT_CRITICAL(&mux);
  wheel(l,0,1);wheel(r,2,3);delay(2);
}

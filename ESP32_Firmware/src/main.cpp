#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>

const int CS_PIN = 5;
const int Button = 13;
const int Probe = 27;

ArduCAM myCam(OV5642, CS_PIN);

unsigned long lastProbeTime = 0;
const unsigned long probeInterval = 100; // Voltage read every 100ms

void setup() {
  pinMode(Button, INPUT_PULLDOWN); 
  pinMode(Probe, INPUT); 

  Serial.begin(921600);
  delay(1000);
  Serial.println(F("ARDUCAM CHECKS"));

  Wire.begin(21, 22);
  Wire.setClock(100000);
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  // RESET 
  myCam.write_reg(0x07, 0x80);
  delay(100);
  myCam.write_reg(0x07, 0x00);
  delay(100);

  // Check for SPI
  while (1) {
    myCam.write_reg(ARDUCHIP_TEST1, 0x55);
    if (myCam.read_reg(ARDUCHIP_TEST1) != 0x55) {
      Serial.println(F("SPI Error :( )"));
      delay(1000);
    } else {
      Serial.println(F("SPI OK :) "));
      break;
    }
  }

  // Check for OV5642 Chip
  while (1) {
    uint8_t vid, pid;
    myCam.rdSensorReg16_8(OV5642_CHIPID_HIGH, &vid);
    myCam.rdSensorReg16_8(OV5642_CHIPID_LOW, &pid);

    Serial.printf("VID: 0x%02X  PID: 0x%02X\n", vid, pid);

    if ((vid == 0x56) && (pid == 0x42)) {
      Serial.println(F("OV5642 detected"));
      break;
    } else {
      Serial.println(F("No connection :( )"));
      delay(1000);
    }
  }

  // Initialize camera
  myCam.set_format(JPEG);
  myCam.InitCAM();
  myCam.write_reg(ARDUCHIP_TIM, VSYNC_LEVEL_MASK);
  myCam.OV5642_set_JPEG_size(OV5642_1280x960);
  myCam.clear_fifo_flag();

  Serial.println(F("All Clear"));
}

void loop() {
//Voltage is read if one second has passed
  if (millis() - lastProbeTime >= probeInterval) {
    lastProbeTime = millis(); 
    
    // Take 20 readings rapidly and average them
    long totalMilliVolts = 0;
    for (int i = 0; i < 20; i++) {
      totalMilliVolts += analogReadMilliVolts(Probe); 
    }
    float avgMilliVolts = totalMilliVolts / 20.0;
    
    // Convert millivolts to regular Volts
    float pinVoltage = avgMilliVolts / 1000.0;
    
    // Reverse the voltage divider  - we scaled voltage dowon previously in the actualy circuit, so the reading needs to be rescaled here
    float calibrationFactor = 2.04; 
    float actualVoltage = pinVoltage * calibrationFactor;
    
    Serial.printf("Probe Voltage: %.2f V\n", actualVoltage);
  }

  // Camerca button logic
  if (digitalRead(Button) == HIGH) {
    delay(50);
    
    Serial.println(F("Starting capture..."));
    myCam.clear_fifo_flag();
    myCam.start_capture();
    
    while (!myCam.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
      delay(1);
    }
    
    uint32_t length = myCam.read_fifo_length();
    
    if (length == 0) {
      Serial.println(F("Error: FIFO length is 0"));
    } else {
      Serial.printf("Image size: %d bytes\n", length);

      digitalWrite(CS_PIN, LOW);
      myCam.set_fifo_burst();
      
      for (uint32_t i = 0; i < length; i++) {
        uint8_t imageByte = SPI.transfer(0x00);
        Serial.write(imageByte);
      }
      
      digitalWrite(CS_PIN, HIGH);
      myCam.clear_fifo_flag();
      
      Serial.println(F("\nPicture Taken and Read!"));
    }

    while (digitalRead(Button) == HIGH) {
      delay(10);
    }
  }
}
/*
  Using the BNO080 IMU
  By: Nathan Seidle
  SparkFun Electronics
  Date: December 21st, 2017
  License: This code is public domain but you buy me a beer if you use this and we meet someday (Beerware license).

  Feel like supporting our work? Buy a board from SparkFun!
  https://www.sparkfun.com/products/14586

  This example shows how to output the i/j/k/real parts of the rotation vector.
  https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

  It takes about 1ms at 400kHz I2C to read a record from the sensor, but we are polling the sensor continually
  between updates from the sensor. Use the interrupt pin on the BNO080 breakout to avoid polling.

  Hardware Connections:
  Attach the Qwiic Shield to your Arduino/Photon/ESP32 or other
  Plug the sensor onto the shield
  Serial.print it out at 115200 baud to serial monitor.
*/

#include <Wire.h>
#include "SparkFun_BNO080_Arduino_Library.h"
BNO080 myIMU1; //Open I2C ADR jumper - goes to address 0x4B
BNO080 myIMU2; //Closed I2C ADR jumper - goes to address 0x4A
bool initi1=false;
bool initi2=false;
float roll1_init,pitch1_init;
float roll2_init,pitch2_init;
float roll1, pitch1, yaw1; 
float roll2, pitch2, yaw2; 

unsigned long lastReport=millis()-2000;
int reportFreq = 100; //ms

void setup()
{
  Serial.begin(115200);
  //Serial.println();
  //Serial.println("BNO080 Read Example");
  Wire.begin();
  Wire.setClock(400000); //Increase I2C data rate to 400kHz

  //When a large amount of time has passed since we last polled the sensors
  //they can freeze up. To un-freeze it is easiest to power cycle the sensor.

  //Start 2 sensors
  if (myIMU1.begin(0x4A) == false)
  {
    Serial.println("First BNO080 not detected with I2C ADR jumper open. Check your jumpers and the hookup guide. Freezing...");
    while(1);
  }

  if (myIMU2.begin(0x4B) == false)
  {
    Serial.println("Second BNO080 not detected with I2C ADR jumper closed. Check your jumpers and the hookup guide. Freezing...");
    while(1);
  }

  myIMU1.enableRotationVector(50); //Send data update every 50ms
  myIMU2.enableRotationVector(50); //Send data update every 50ms
  //Serial.println(F("Rotation vector enabled"));
 // Serial.println(F("Output in form i, j, k, real, accuracy"));
 delay(5000);


}

void loop()
{
  //Look for reports from the IMU

  if (myIMU1.dataAvailable() == true)
  {
    roll1 = (myIMU1.getRoll()) * 180.0 / PI; // Convert roll to degrees
    pitch1 = (myIMU1.getPitch()) * 180.0 / PI; // Convert pitch to degrees
    yaw1 = (myIMU1.getYaw()) * 180.0 / PI; // Convert yaw / heading to degrees
  }

  if (myIMU2.dataAvailable() == true)
  {
    roll2 = (myIMU2.getRoll()) * 180.0 / PI; // Convert roll to degrees
    pitch2 = (myIMU2.getPitch()) * 180.0 / PI; // Convert pitch to degrees
    yaw2 = (myIMU2.getYaw()) * 180.0 / PI; // Convert yaw / heading to degrees
  
  }
  if (reportFreq< millis()-lastReport)
  {
    
    lastReport = millis();
    Serial.flush();

    Serial.print(lastReport);Serial.print(",\tR1");
    Serial.print(roll1-roll1_init, 1);Serial.print(",\tP1");
    Serial.print(pitch1-pitch1_init, 1);Serial.print(",\tY1");
    Serial.print(yaw1, 1);Serial.print(",");
    
    //Serial.print(roll2, 1);Serial.print(",");
    //Serial.print(pitch2, 1);Serial.print(",");
    //Serial.print(yaw2, 1);
    Serial.println(",");
  }
  delay(10);
}

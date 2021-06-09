/*
  Using the BNO080 IMU
  Return Quaternion to serial
  */

#include <Wire.h>

#include "SparkFun_BNO080_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_BNO080
BNO080 myIMU1; //Open I2C ADR jumper - goes to address 0x4B
BNO080 myIMU2; //Closed I2C ADR jumper - goes to address 0x4A
float quatI1,quatJ1,quatK1,quatReal1,quatRadianAccuracy1;
float quatI1_initi,quatJ1_initi,quatK1_initi,quatReal1_initi;
float quatI2,quatJ2,quatK2,quatReal2,quatRadianAccuracy2;
unsigned long lastReport=millis()-2000;
int reportFreq = 10; //ms



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
  if (myIMU1.begin(0x4B) == false)
  {
    Serial.println("First BNO080 not detected with I2C ADR jumper open. Check your jumpers and the hookup guide. Freezing...");
    while(1);
  }

  if (myIMU2.begin(0x4A) == false)
  {
    Serial.println("Second BNO080 not detected with I2C ADR jumper closed. Check your jumpers and the hookup guide. Freezing...");
    while(1);
  }
  /*
    Serial.println(F("Rotation vector enabled"));
  Serial.println(F("Output in form i, j, k, real, accuracy"));
  */
  myIMU1.enableRotationVector(50); //Send data update every 50ms
  myIMU2.enableRotationVector(50); //Send data update every 50ms
  /*
   while (myIMU1.dataAvailable() == false || quatI1_initi==0)
  {
    quatI1_initi = myIMU1.getQuatI();
    quatJ1_initi = myIMU1.getQuatJ();
    quatK1_initi = myIMU1.getQuatK();
  }
  */

}

void loop()
{
  //Look for reports from the IMU
  if (myIMU1.dataAvailable() == true)
  {
    quatI1 = myIMU1.getQuatI();
    quatJ1 = myIMU1.getQuatJ();
    quatK1 = myIMU1.getQuatK();
    quatReal1 = myIMU1.getQuatReal();
  }

  if (myIMU2.dataAvailable() == true)
  {
    quatI2 = myIMU2.getQuatI();
    quatJ2 = myIMU2.getQuatJ();
    quatK2 = myIMU2.getQuatK();
    quatReal2 = myIMU2.getQuatReal();
  }
  if (reportFreq< millis()-lastReport)
  {

    lastReport = millis();

    Serial.print(quatI1);Serial.print(F(","));
    Serial.print(quatJ1);Serial.print(F(","));
    Serial.print(quatK1);Serial.print(F(","));
    Serial.print(quatReal1);Serial.print(F(","));

    Serial.print(quatI2);Serial.print(F(","));
    Serial.print(quatJ2);Serial.print(F(","));
    Serial.print(quatK2);Serial.print(F(","));
    Serial.print(quatReal2);


    Serial.println();
  }
  delay(10);
}

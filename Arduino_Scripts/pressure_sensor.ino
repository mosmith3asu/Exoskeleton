/*Hardware Hookup:
Mux Breakout ----------- Arduino
     S0 ------------------- 4
     S1 ------------------- 5
     SIG ------------------ A0
    VCC ------------------- 5V
    GND ------------------- GND
    this connection is for C0 and C1
*/
const int S00=50;
const int S01=51;
const int S02=52;
const int S03=53;
int t=0;
int i=0;
String IMU_Data;
//const int SIG=A0; // Connect common (Z) to A0 (analog input)
int array1[18];
int array2[18];
int ini_1[18];
int ini_2[18];
int ini_3[18];

String serialBuffer;
int serialDelay = 100;

void setup() 
{
  Serial.begin(115200); // Initialize the serial port
   // Serial1.begin(115200); //arduino Uno serial port (1)
//  while (!Serial) {
//    ; // wait for serial port to connect. Needed for native USB
//  }
  // Set up the select pins as outputs:
  pinMode(S01, OUTPUT);
  pinMode(S02, OUTPUT);
  pinMode(S00, OUTPUT);
  pinMode(S03, OUTPUT);
 // pinMode(SIG, INPUT); // Set up Z as an input
   for (int i=0;i<=17;i++){
      array1[i]=0;
      array2[i]=0;
      ini_1[i]=0;
      ini_2[i]=0;
      ini_3[i]=0;
}
}
// I am the big daddy of coding.
void loop() 
{                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
int i=0;
        digitalWrite(S00,LOW);
    digitalWrite(S01,LOW);
    digitalWrite(S02,LOW);
    digitalWrite(S03,LOW);  
    //digitalWrite(S04,HIGH);                       
    for (i=1;i<=50;i++){
      array1[0]=array1[0]+analogRead(A0);
      array1[12]=array1[12]+analogRead(A1);
      //var3=var3+analogRead(A2);
      }
       digitalWrite(S00,HIGH);
    digitalWrite(S01,LOW);
    digitalWrite(S02,LOW);
    digitalWrite(S03,LOW);                         
    for (i=1;i<=50;i++){
      array1[1]=array1[1]+analogRead(A0);
      array1[13]=array1[13]+analogRead(A1);
      array2[0]=array2[0]+analogRead(A2);
     }
         digitalWrite(S00,LOW);
    digitalWrite(S01,HIGH);
    digitalWrite(S02,LOW);
    digitalWrite(S03,LOW);                         
    for (i=1;i<=50;i++){
      array1[2]=array1[2]+analogRead(A0);
     // array1[14]=array1[14]+analogRead(A1);
      array2[1]=array2[1]+analogRead(A2);
      }
     digitalWrite(S00,HIGH);
    digitalWrite(S01,HIGH);
    digitalWrite(S02,LOW);
    digitalWrite(S03,LOW);                         
    for (i=1;i<=50;i++){
      array1[3]=array1[3]+analogRead(A0);
      //array1[15]=array1[15]+analogRead(A1);
      //array2[2]=array2[2]+analogRead(A2);
      }
    digitalWrite(S00,LOW);
    digitalWrite(S01,LOW);
    digitalWrite(S02,HIGH);
    digitalWrite(S03,LOW);                         
    for (i=1;i<=50;i++){
      array1[4]=array1[4]+analogRead(A0);
      array1[16]=array1[16]+analogRead(A1);
     // array2[3]=array2[3]+analogRead(A2);
      }
       digitalWrite(S00,HIGH);
    digitalWrite(S01,LOW);
    digitalWrite(S02,HIGH);
    digitalWrite(S03,LOW);                         
    for (i=1;i<=50;i++){
      array1[5]=array1[5]+analogRead(A0);
      array1[17]=array1[17]+analogRead(A1);
     array2[4]=array2[4]+analogRead(A2);
      }
       digitalWrite(S00,LOW);
    digitalWrite(S01,HIGH);
    digitalWrite(S02,HIGH);
    digitalWrite(S03,LOW);                         
    for (i=1;i<=50;i++){
      array1[6]=array1[6]+analogRead(A0);
     array2[5]=array2[5]+analogRead(A2);
      }
       digitalWrite(S00,HIGH);
    digitalWrite(S01,HIGH);
    digitalWrite(S02,HIGH);
    digitalWrite(S03,LOW);                         
    for (i=1;i<=50;i++){
      array1[7]=array1[7]+analogRead(A0);
     array2[6]=array2[6]+analogRead(A2);
      }
        digitalWrite(S00,LOW);
    digitalWrite(S01,LOW);
    digitalWrite(S02,LOW);
    digitalWrite(S03,HIGH);                         
    for (i=1;i<=50;i++){
      array1[8]=array1[8]+analogRead(A0);
     array1[14]=array1[14]+analogRead(A1);
     array2[7]=array2[7]+analogRead(A2);
      }
      digitalWrite(S00,HIGH);
    digitalWrite(S01,LOW);
    digitalWrite(S02,LOW);
    digitalWrite(S03,HIGH);                         
    for (i=1;i<=50;i++){
      array1[9]=array1[9]+analogRead(A0);
     array1[15]=array1[15]+analogRead(A1);
     array2[8]=array2[8]+analogRead(A2);
      }
      digitalWrite(S00,LOW);
    digitalWrite(S01,HIGH);
    digitalWrite(S02,LOW);
    digitalWrite(S03,HIGH);                         
    for (i=1;i<=50;i++){
      array1[10]=array1[10]+analogRead(A0);
     array2[9]=array2[9]+analogRead(A2);
      }
      digitalWrite(S00,HIGH);
    digitalWrite(S01,HIGH);
    digitalWrite(S02,LOW);
    digitalWrite(S03,HIGH);                         
    for (i=1;i<=50;i++){
      array1[11]=array1[11]+analogRead(A0);
    // array2[10]=array2[10]+analogRead(A2);
      }
//       digitalWrite(S00,LOW);
//    digitalWrite(S01,LOW);
//    digitalWrite(S02,HIGH);
//    digitalWrite(S03,HIGH);                         
//    for (i=1;i<=50;i++){
//     array2[11]=array2[11]+analogRead(A2);
//      }
//       digitalWrite(S00,HIGH);
//    digitalWrite(S01,LOW);
//    digitalWrite(S02,HIGH);
//    digitalWrite(S03,HIGH);                         
//    for (i=1;i<=50;i++){
//     array2[12]=array2[12]+analogRead(A2);
//      }
      digitalWrite(S00,LOW);
    digitalWrite(S01,HIGH);
    digitalWrite(S02,HIGH);
    digitalWrite(S03,HIGH);                         
    for (i=1;i<=50;i++){
     array2[3]=array2[3]+analogRead(A2);
      }
      digitalWrite(S00,HIGH);
    digitalWrite(S01,HIGH);
    digitalWrite(S02,HIGH);
    digitalWrite(S03,HIGH);                         
    for (i=1;i<=50;i++){
     array2[2]=array2[2]+analogRead(A2);
      }
    for (i=0;i<=17;i++){
      array1[i]=array1[i]/50;
      array2[i]=array2[i]/50;};
 //SerialUSB.println(t);
if (t<51) {
  for (i=0;i<=17;i++){
  ini_1[i]=array1[i]+ini_1[i];
  ini_2[i]=array2[i]+ini_2[i];}
  t=t+1;
  //SerialUSB.println(t);
};
 if (t>50) {
 for (i=0;i<=17;i++){
 array1[i]=array1[i]-ini_1[i]/51;
 array2[i]=array2[i]-ini_2[i]/51;  
 }};
   serialBuffer = "";
   for (i=0;i<=17;i++){
      //Serial.print(array1[i]);
      //Serial.print(", ");}
      serialBuffer.concat(array1[i]);
      serialBuffer.concat(",");}
   for (i=0;i<=9;i++){
      //Serial.print(array2[i]);
      //Serial.print(", ");}
      serialBuffer.concat(array2[i]);
      serialBuffer.concat(",");}
    
    
//if (Serial1.available() > 0)               //if there is serial data in buffer 
//   {
//    IMU_Data= Serial1.readStringUntil('\n');  //Read recieved string from serial port 1
//    Serial.println(IMU_Data);                 //Print IMU Data from Uno to serial monitor
//  }
         for (i=0;i<=17;i++){
      array1[i]=0;
      array2[i]=0;};
      Serial.println(serialBuffer);
      delay(serialDelay);
    };

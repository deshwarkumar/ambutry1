#include <LiquidCrystal.h>
#include <SoftwareSerial.h>
SoftwareSerial mySerial(2,3);
#include <TinyGPS.h>
TinyGPS gps;
float flat=0, flon=0;
#define buz 4
const int rs = 8, en = 9, d4 = 10, d5 = 11, d6 = 12, d7 = 13;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

int tr=6;
int ec=7;
int cnt=0;
int ss=0;
void read_gps()
{
  bool newData = false;
  unsigned long chars;
  unsigned short sentences, failed;
  for (unsigned long start = millis(); millis() - start < 1000;)
  {
    while (mySerial.available())
    {
      char c = mySerial.read();
      if (gps.encode(c)) 
        newData = true;
    }
  }

  if (newData)
  {
    
    unsigned long age;
    gps.f_get_position(&flat, &flon, &age);

  }
}



void setup() {
  
  Serial.begin(9600); 
  mySerial.begin(9600); 
  pinMode(buz,OUTPUT);
  digitalWrite(buz,0);
  lcd.begin(16, 2);
  lcd.print("   WELCOME");
  pinMode(tr,OUTPUT);
  pinMode(ec,INPUT);

 
  delay(1000);
  
  
  
}

void loop() {



  
read_gps();

digitalWrite(tr,1);
delayMicroseconds(10);
digitalWrite(tr,0);
int dst=pulseIn(ec,1)/58.2;

if (Serial.available()) 
{
int rcv = Serial.read();
if(rcv=='1')
{

digitalWrite(buz,1);
ss=1; 
}

}


lcd.clear();
lcd.print("LT:"+String(flat,4) + " D:"+String(dst));
lcd.setCursor(0,1);
lcd.print("LG:"+String(flon,4));


Serial.println(String(dst) + ","+String(flat,4)+","+String(flon,4));
 if(ss==1)
 cnt=cnt+1;
 if(cnt>10)
 {
 cnt=0;
 ss=0;
 digitalWrite(buz,0);
 delay(100);
 }
 }

 

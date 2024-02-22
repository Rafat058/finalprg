
#include <PZEM004Tv30.h>

#define RX_PIN 2
#define TX_PIN 3

PZEM004Tv30 pzem(RX_PIN, TX_PIN); // Using Hardware Serial

void setup() {
  Serial.begin(9600);
}

void loop() {
  float voltage = pzem.voltage();
  float current = pzem.current();
  float power = pzem.power();

  // Send sensor data to the PC
  Serial.print(voltage);
  Serial.print(",");
  Serial.print(current);
  Serial.print(",");
  Serial.println(power);

  delay(1000); // Adjust delay as needed
}

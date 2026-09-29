const int RELAY = 7;

void setup() {
    pinMode(RELAY, OUTPUT);

    digitalWrite(RELAY, HIGH);

    Serial.begin(115200);

    Serial.println("Ready");
}

void loop() {

    if (Serial.available()) {

        char c = toupper(Serial.read());

        if (c == 'Y') {
            digitalWrite(RELAY, LOW);
            Serial.println("ON");
        }

        if (c == 'N') {
            digitalWrite(RELAY, HIGH);
            Serial.println("OFF");
        }
    }
}
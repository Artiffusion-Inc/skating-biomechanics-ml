const int packetHeader = 0x55;
const int typeAccel = 0x51;
const int typeGyro = 0x52;
const int typeQuat = 0x59;
const int typeAngle = 0x53;
const int typeBattery = 0x5C;
const int packetLength = 11;

const double scaleAcc = 16.0 / 32768.0;
const double scaleGyro = 2000.0 / 32768.0;
const double scaleQuat = 1.0 / 32768.0;
const double scaleAngle = 180.0 / 32768.0;

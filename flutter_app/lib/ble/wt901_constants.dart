const int PACKET_HEADER = 0x55;
const int TYPE_ACCEL = 0x51;
const int TYPE_GYRO = 0x52;
const int TYPE_ANGLE = 0x53;
const int TYPE_QUAT = 0x59;
const int PACKET_LENGTH = 11;

const double SCALE_ACC = 16.0 / 32768.0;
const double SCALE_GYRO = 2000.0 / 32768.0;
const double SCALE_QUAT = 1.0 / 32768.0;

import 'dart:typed_data';
import 'wt901_constants.dart';

enum WT901PacketType { accelerometer, gyroscope, angle, quaternion, unknown }

class WT901Packet {
  final WT901PacketType type;
  final double? accX, accY, accZ;
  final double? gyroX, gyroY, gyroZ;
  final double? quatW, quatX, quatY, quatZ;

  WT901Packet({
    required this.type,
    this.accX, this.accY, this.accZ,
    this.gyroX, this.gyroY, this.gyroZ,
    this.quatW, this.quatX, this.quatY, this.quatZ,
  });
}

class WT901Parser {
  static WT901Packet? parse(List<int> raw) {
    if (raw.length < PACKET_LENGTH) return null;
    if (raw[0] != PACKET_HEADER) return null;

    final typeByte = raw[1];
    WT901PacketType type;
    switch (typeByte) {
      case TYPE_ACCEL: type = WT901PacketType.accelerometer; break;
      case TYPE_GYRO: type = WT901PacketType.gyroscope; break;
      case TYPE_ANGLE: type = WT901PacketType.angle; break;
      case TYPE_QUAT: type = WT901PacketType.quaternion; break;
      default: type = WT901PacketType.unknown;
    }

    final buffer = Uint8List.fromList(raw.sublist(0, PACKET_LENGTH)).buffer;
    final data = ByteData.view(buffer);

    double readInt16Scaled(int offset, double scale) {
      return data.getInt16(offset, Endian.little) * scale;
    }

    double? accX, accY, accZ;
    double? gyroX, gyroY, gyroZ;
    double? quatW, quatX, quatY, quatZ;

    if (type == WT901PacketType.accelerometer) {
      accX = readInt16Scaled(2, SCALE_ACC);
      accY = readInt16Scaled(4, SCALE_ACC);
      accZ = readInt16Scaled(6, SCALE_ACC);
    } else if (type == WT901PacketType.gyroscope) {
      gyroX = readInt16Scaled(2, SCALE_GYRO);
      gyroY = readInt16Scaled(4, SCALE_GYRO);
      gyroZ = readInt16Scaled(6, SCALE_GYRO);
    } else if (type == WT901PacketType.quaternion) {
      quatW = readInt16Scaled(2, SCALE_QUAT);
      quatX = readInt16Scaled(4, SCALE_QUAT);
      quatY = readInt16Scaled(6, SCALE_QUAT);
      quatZ = readInt16Scaled(8, SCALE_QUAT);
    }

    return WT901Packet(
      type: type,
      accX: accX, accY: accY, accZ: accZ,
      gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ,
      quatW: quatW, quatX: quatX, quatY: quatY, quatZ: quatZ,
    );
  }
}

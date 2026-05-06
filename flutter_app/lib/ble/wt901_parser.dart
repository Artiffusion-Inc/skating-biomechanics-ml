import 'dart:typed_data';
import 'wt901_constants.dart';

enum WT901PacketType { accelerometer, gyroscope, quaternion, unknown }

class WT901Packet {
  final WT901PacketType type;
  final double? accX, accY, accZ;
  final double? gyroX, gyroY, gyroZ;
  final double? quatW, quatX, quatY, quatZ;

  const WT901Packet({
    required this.type,
    this.accX, this.accY, this.accZ,
    this.gyroX, this.gyroY, this.gyroZ,
    this.quatW, this.quatX, this.quatY, this.quatZ,
  });
}

class WT901Parser {
  static int _computeChecksum(List<int> raw) {
    var sum = 0;
    for (var i = 0; i < packetLength - 1; i++) {
      sum += raw[i];
    }
    return sum & 0xFF;
  }

  static double _readInt16Scaled(ByteData data, int offset, double scale) {
    return data.getInt16(offset, Endian.little) * scale;
  }

  static WT901Packet? parse(List<int> raw) {
    if (raw.length < packetLength) return null;
    if (raw[0] != packetHeader) return null;

    final checksum = _computeChecksum(raw);
    if (raw[packetLength - 1] != checksum) return null;

    final typeByte = raw[1];
    WT901PacketType type;
    switch (typeByte) {
      case typeAccel: type = WT901PacketType.accelerometer; break;
      case typeGyro: type = WT901PacketType.gyroscope; break;
      case typeQuat: type = WT901PacketType.quaternion; break;
      default: type = WT901PacketType.unknown;
    }

    final data = ByteData.sublistView(
      Uint8List.fromList(raw.sublist(0, packetLength)),
    );

    double? accX, accY, accZ;
    double? gyroX, gyroY, gyroZ;
    double? quatW, quatX, quatY, quatZ;

    if (type == WT901PacketType.accelerometer) {
      accX = _readInt16Scaled(data, 2, scaleAcc);
      accY = _readInt16Scaled(data, 4, scaleAcc);
      accZ = _readInt16Scaled(data, 6, scaleAcc);
    } else if (type == WT901PacketType.gyroscope) {
      gyroX = _readInt16Scaled(data, 2, scaleGyro);
      gyroY = _readInt16Scaled(data, 4, scaleGyro);
      gyroZ = _readInt16Scaled(data, 6, scaleGyro);
    } else if (type == WT901PacketType.quaternion) {
      quatW = _readInt16Scaled(data, 2, scaleQuat);
      quatX = _readInt16Scaled(data, 4, scaleQuat);
      quatY = _readInt16Scaled(data, 6, scaleQuat);
      quatZ = _readInt16Scaled(data, 8, scaleQuat);
    }

    return WT901Packet(
      type: type,
      accX: accX, accY: accY, accZ: accZ,
      gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ,
      quatW: quatW, quatX: quatX, quatY: quatY, quatZ: quatZ,
    );
  }
}

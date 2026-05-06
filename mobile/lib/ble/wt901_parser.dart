import 'dart:typed_data';
import 'wt901_constants.dart';

enum WT901PacketType { accelerometer, gyroscope, quaternion, angle, battery, unknown }

class WT901Packet {
  final WT901PacketType type;
  final double? accX, accY, accZ;
  final double? gyroX, gyroY, gyroZ;
  final double? angleX, angleY, angleZ;
  final double? quatW, quatX, quatY, quatZ;
  final double? battery; // voltage in V

  const WT901Packet({
    required this.type,
    this.accX, this.accY, this.accZ,
    this.gyroX, this.gyroY, this.gyroZ,
    this.angleX, this.angleY, this.angleZ,
    this.quatW, this.quatX, this.quatY, this.quatZ,
    this.battery,
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

    final packet = raw.sublist(0, packetLength);
    final checksum = _computeChecksum(packet);
    if (packet[packetLength - 1] != checksum) return null;

    final typeByte = packet[1];
    WT901PacketType type;
    switch (typeByte) {
      case typeAccel: type = WT901PacketType.accelerometer; break;
      case typeGyro: type = WT901PacketType.gyroscope; break;
      case typeAngle: type = WT901PacketType.angle; break;
      case typeQuat: type = WT901PacketType.quaternion; break;
      case typeBattery: type = WT901PacketType.battery; break;
      default: type = WT901PacketType.unknown;
    }

    final data = ByteData.sublistView(
      Uint8List.fromList(packet),
    );

    double? accX, accY, accZ;
    double? gyroX, gyroY, gyroZ;
    double? angleX, angleY, angleZ;
    double? quatW, quatX, quatY, quatZ;
    double? battery;

    if (type == WT901PacketType.accelerometer) {
      accX = _readInt16Scaled(data, 2, scaleAcc);
      accY = _readInt16Scaled(data, 4, scaleAcc);
      accZ = _readInt16Scaled(data, 6, scaleAcc);
    } else if (type == WT901PacketType.gyroscope) {
      gyroX = _readInt16Scaled(data, 2, scaleGyro);
      gyroY = _readInt16Scaled(data, 4, scaleGyro);
      gyroZ = _readInt16Scaled(data, 6, scaleGyro);
    } else if (type == WT901PacketType.angle) {
      angleX = _readInt16Scaled(data, 2, scaleAngle);
      angleY = _readInt16Scaled(data, 4, scaleAngle);
      angleZ = _readInt16Scaled(data, 6, scaleAngle);
    } else if (type == WT901PacketType.quaternion) {
      quatW = _readInt16Scaled(data, 2, scaleQuat);
      quatX = _readInt16Scaled(data, 4, scaleQuat);
      quatY = _readInt16Scaled(data, 6, scaleQuat);
      quatZ = _readInt16Scaled(data, 8, scaleQuat);
    } else if (type == WT901PacketType.battery) {
      battery = data.getInt16(2, Endian.little) / 100.0;
    }

    return WT901Packet(
      type: type,
      accX: accX, accY: accY, accZ: accZ,
      gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ,
      angleX: angleX, angleY: angleY, angleZ: angleZ,
      quatW: quatW, quatX: quatX, quatY: quatY, quatZ: quatZ,
      battery: battery,
    );
  }
}

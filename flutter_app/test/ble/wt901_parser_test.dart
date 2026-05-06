import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/ble/wt901_constants.dart';
import 'package:edgesense_capture/ble/wt901_parser.dart';

List<int> _withChecksum(List<int> payload) {
  // Pad payload to exactly packetLength bytes (last byte is checksum placeholder)
  final packet = List<int>.filled(packetLength, 0x00);
  for (var i = 0; i < payload.length && i < packetLength; i++) {
    packet[i] = payload[i];
  }
  var sum = 0;
  for (var i = 0; i < packetLength - 1; i++) {
    sum += packet[i];
  }
  packet[packetLength - 1] = sum & 0xFF;
  return packet;
}

void main() {
  group('WT901Parser', () {
    test('returns null for invalid header', () {
      final packet = [0x00, typeAccel, ...List.filled(packetLength - 2, 0x00)];
      expect(WT901Parser.parse(_withChecksum(packet)), isNull);
    });

    test('returns null for short packet', () {
      final packet = [packetHeader, typeAccel, 0x00];
      expect(WT901Parser.parse(packet), isNull);
    });

    test('returns null for invalid checksum', () {
      final packet = [packetHeader, typeAccel, ...List.filled(packetLength - 3, 0x00), 0xFF];
      expect(WT901Parser.parse(packet), isNull);
    });

    test('parses accelerometer packet with zero payload', () {
      final packet = _withChecksum([packetHeader, typeAccel, ...List.filled(packetLength - 2, 0x00)]);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.accelerometer);
      expect(result.accX, closeTo(0.0, 0.001));
      expect(result.accY, closeTo(0.0, 0.001));
      expect(result.accZ, closeTo(0.0, 0.001));
    });

    test('parses gyroscope packet', () {
      // gyro X=1, Y=2, Z=3 (raw int16 little-endian)
      final payload = [
        packetHeader, typeGyro,
        0x01, 0x00, // X = 1
        0x02, 0x00, // Y = 2
        0x03, 0x00, // Z = 3
        0x00, // padding
      ];
      final packet = _withChecksum(payload);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.gyroscope);
      expect(result.gyroX, closeTo(1.0 * scaleGyro, 0.001));
      expect(result.gyroY, closeTo(2.0 * scaleGyro, 0.001));
      expect(result.gyroZ, closeTo(3.0 * scaleGyro, 0.001));
    });

    test('parses angle packet', () {
      final payload = [
        packetHeader, typeAngle,
        0x64, 0x00, // X = 100
        0x32, 0x00, // Y = 50
        0x19, 0x00, // Z = 25
        0x00, // padding
      ];
      final packet = _withChecksum(payload);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.angle);
      expect(result.angleX, closeTo(100.0 * scaleAngle, 0.001));
      expect(result.angleY, closeTo(50.0 * scaleAngle, 0.001));
      expect(result.angleZ, closeTo(25.0 * scaleAngle, 0.001));
    });

    test('parses quaternion packet with all components', () {
      final payload = [
        packetHeader, typeQuat,
        0x64, 0x00, // W = 100
        0x32, 0x00, // X = 50
        0x19, 0x00, // Y = 25
        0x0A, 0x00, // Z = 10
      ];
      final packet = _withChecksum(payload);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.quaternion);
      expect(result.quatW, closeTo(100.0 * scaleQuat, 0.001));
      expect(result.quatX, closeTo(50.0 * scaleQuat, 0.001));
      expect(result.quatY, closeTo(25.0 * scaleQuat, 0.001));
      expect(result.quatZ, closeTo(10.0 * scaleQuat, 0.001));
    });

    test('parses unknown type packet', () {
      final packet = _withChecksum([packetHeader, 0xFF, ...List.filled(packetLength - 2, 0x00)]);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.unknown);
      expect(result.accX, isNull);
      expect(result.gyroX, isNull);
      expect(result.angleX, isNull);
      expect(result.quatW, isNull);
    });

    test('checksum rollover', () {
      // Payload where sum of first 10 bytes exceeds 255
      final payload = [packetHeader, typeAccel, ...List.filled(packetLength - 2, 0xFF)];
      final packet = _withChecksum(payload);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.accelerometer);
    });

    test('scale factors produce physical units for accelerometer', () {
      // Z at 1g: raw = 32768 / 16 = 2048 = 0x0800 little-endian
      final payload = [
        packetHeader, typeAccel,
        0x00, 0x00, // X
        0x00, 0x00, // Y
        0x00, 0x08, // Z = 2048
        0x00, // padding
      ];
      final packet = _withChecksum(payload);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.accZ, closeTo(1.0, 0.01));
    });

    test('handles negative int16 values', () {
      // -1 = 0xFFFF little-endian
      final payload = [
        packetHeader, typeAccel,
        0xFF, 0xFF, // X = -1
        0x00, 0x00, // Y
        0x00, 0x00, // Z
        0x00, // padding
      ];
      final packet = _withChecksum(payload);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.accX, closeTo(-1.0 * scaleAcc, 0.001));
    });

    test('handles boundary int16 max positive', () {
      // 32767 = 0x7FFF little-endian
      final payload = [
        packetHeader, typeAccel,
        0xFF, 0x7F, // X = 32767
        0x00, 0x00, // Y
        0x00, 0x00, // Z
        0x00, // padding
      ];
      final packet = _withChecksum(payload);
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.accX, closeTo(32767.0 * scaleAcc, 0.001));
    });

    test('truncates packet longer than packetLength', () {
      final payload = [packetHeader, typeAccel, ...List.filled(packetLength - 2, 0x00)];
      final packet = _withChecksum(payload);
      final longPacket = [...packet, 0xFF, 0xFF]; // extra bytes
      final result = WT901Parser.parse(longPacket);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.accelerometer);
    });
  });
}

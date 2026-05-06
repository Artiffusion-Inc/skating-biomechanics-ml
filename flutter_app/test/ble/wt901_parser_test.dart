import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/ble/wt901_constants.dart';
import 'package:edgesense_capture/ble/wt901_parser.dart';

void main() {
  group('WT901Parser', () {
    test('returns null for invalid header', () {
      final packet = [0x00, 0x51, ...List.filled(9, 0x00)];
      expect(WT901Parser.parse(packet), isNull);
    });

    test('returns null for short packet', () {
      final packet = [0x55, 0x51, 0x00];
      expect(WT901Parser.parse(packet), isNull);
    });

    test('parses accelerometer packet', () {
      // 0x55 0x51 + 8 bytes payload (all zeros) + checksum
      final packet = [0x55, 0x51, ...List.filled(9, 0x00)];
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.accelerometer);
      expect(result.accX, closeTo(0.0, 0.001));
      expect(result.accY, closeTo(0.0, 0.001));
      expect(result.accZ, closeTo(0.0, 0.001));
    });

    test('parses quaternion packet', () {
      final packet = [0x55, 0x59, ...List.filled(9, 0x00)];
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.quaternion);
      expect(result.quatW, closeTo(0.0, 0.001));
    });

    test('scale factors produce physical units', () {
      // Accelerometer Z at 1g: raw = 2048 (32768/16)
      final packet = [
        0x55, 0x51, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00
      ];
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.accZ, closeTo(1.0, 0.01));
    });
  });
}

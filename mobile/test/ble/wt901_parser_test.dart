import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/ble/wt901_parser.dart';
import 'package:edgesense_capture/ble/wt901_constants.dart';

void main() {
  group('WT901Parser', () {
    test(
      'parses concatenated packet by taking only first packetLength bytes',
      () {
        final valid = List<int>.filled(packetLength, 0x55);
        valid[0] = packetHeader;
        valid[1] = typeAccel;
        var sum = 0;
        for (var i = 0; i < packetLength - 1; i++) sum += valid[i];
        valid[packetLength - 1] = sum & 0xFF;

        final garbage = List<int>.filled(packetLength, 0xFF);
        final concatenated = [...valid, ...garbage];

        final result = WT901Parser.parse(concatenated);
        expect(result, isNotNull);
        expect(result!.type, WT901PacketType.accelerometer);
      },
    );

    test('returns null for truncated packet', () {
      final truncated = [packetHeader, typeAccel];
      expect(WT901Parser.parse(truncated), isNull);
    });

    test('returns null for wrong checksum', () {
      final raw = List<int>.filled(packetLength, 0x55);
      raw[0] = packetHeader;
      raw[1] = typeAccel;
      raw[packetLength - 1] = 0x00; // wrong checksum
      expect(WT901Parser.parse(raw), isNull);
    });
  });
}

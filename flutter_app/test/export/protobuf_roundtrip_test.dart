import 'package:flutter_test/flutter_test.dart';
import 'package:fixnum/fixnum.dart';
import 'package:edgesense_capture/export/protobuf_gen/imu.pb.dart';

void main() {
  test('IMUSample roundtrip encode/decode covers all fields', () {
    final sample = IMUSample()
      ..relativeTimestampMs = Int64(1234)
      ..accX = 1.0
      ..accY = 2.0
      ..accZ = 3.0
      ..gyroX = 0.1
      ..gyroY = 0.2
      ..gyroZ = 0.3
      ..quatW = 1.0
      ..quatX = 0.0
      ..quatY = 0.0
      ..quatZ = 0.0;

    final bytes = sample.writeToBuffer();
    final decoded = IMUSample()..mergeFromBuffer(bytes);

    expect(decoded.relativeTimestampMs.toInt(), equals(1234));
    expect(decoded.accX, closeTo(1.0, 0.001));
    expect(decoded.accY, closeTo(2.0, 0.001));
    expect(decoded.accZ, closeTo(3.0, 0.001));
    expect(decoded.gyroX, closeTo(0.1, 0.001));
    expect(decoded.gyroY, closeTo(0.2, 0.001));
    expect(decoded.gyroZ, closeTo(0.3, 0.001));
    expect(decoded.quatW, closeTo(1.0, 0.001));
    expect(decoded.quatX, closeTo(0.0, 0.001));
    expect(decoded.quatY, closeTo(0.0, 0.001));
    expect(decoded.quatZ, closeTo(0.0, 0.001));
  });

  test('IMUStream roundtrip encode/decode with 2 samples', () {
    final s1 = IMUSample()
      ..relativeTimestampMs = Int64(100)
      ..accX = 1.0
      ..accY = 2.0
      ..accZ = 3.0
      ..gyroX = 0.1
      ..gyroY = 0.2
      ..gyroZ = 0.3
      ..quatW = 1.0
      ..quatX = 0.0
      ..quatY = 0.0
      ..quatZ = 0.0;

    final s2 = IMUSample()
      ..relativeTimestampMs = Int64(200)
      ..accX = -1.0
      ..accY = -2.0
      ..accZ = -3.0
      ..gyroX = -0.1
      ..gyroY = -0.2
      ..gyroZ = -0.3
      ..quatW = 0.0
      ..quatX = 1.0
      ..quatY = 0.0
      ..quatZ = 0.0;

    final stream = IMUStream()..samples.addAll([s1, s2]);
    final bytes = stream.writeToBuffer();
    final decoded = IMUStream()..mergeFromBuffer(bytes);

    expect(decoded.samples.length, equals(2));

    final d1 = decoded.samples[0];
    expect(d1.relativeTimestampMs.toInt(), equals(100));
    expect(d1.accX, closeTo(1.0, 0.001));
    expect(d1.accY, closeTo(2.0, 0.001));
    expect(d1.accZ, closeTo(3.0, 0.001));
    expect(d1.gyroX, closeTo(0.1, 0.001));
    expect(d1.gyroY, closeTo(0.2, 0.001));
    expect(d1.gyroZ, closeTo(0.3, 0.001));
    expect(d1.quatW, closeTo(1.0, 0.001));
    expect(d1.quatX, closeTo(0.0, 0.001));
    expect(d1.quatY, closeTo(0.0, 0.001));
    expect(d1.quatZ, closeTo(0.0, 0.001));

    final d2 = decoded.samples[1];
    expect(d2.relativeTimestampMs.toInt(), equals(200));
    expect(d2.accX, closeTo(-1.0, 0.001));
    expect(d2.accY, closeTo(-2.0, 0.001));
    expect(d2.accZ, closeTo(-3.0, 0.001));
    expect(d2.gyroX, closeTo(-0.1, 0.001));
    expect(d2.gyroY, closeTo(-0.2, 0.001));
    expect(d2.gyroZ, closeTo(-0.3, 0.001));
    expect(d2.quatW, closeTo(0.0, 0.001));
    expect(d2.quatX, closeTo(1.0, 0.001));
    expect(d2.quatY, closeTo(0.0, 0.001));
    expect(d2.quatZ, closeTo(0.0, 0.001));
  });

  test('IMUSample edge case: all zero values', () {
    final sample = IMUSample()
      ..relativeTimestampMs = Int64(0)
      ..accX = 0.0
      ..accY = 0.0
      ..accZ = 0.0
      ..gyroX = 0.0
      ..gyroY = 0.0
      ..gyroZ = 0.0
      ..quatW = 0.0
      ..quatX = 0.0
      ..quatY = 0.0
      ..quatZ = 0.0;

    final bytes = sample.writeToBuffer();
    final decoded = IMUSample()..mergeFromBuffer(bytes);

    expect(decoded.relativeTimestampMs.toInt(), equals(0));
    expect(decoded.accX, closeTo(0.0, 0.001));
    expect(decoded.accY, closeTo(0.0, 0.001));
    expect(decoded.accZ, closeTo(0.0, 0.001));
    expect(decoded.gyroX, closeTo(0.0, 0.001));
    expect(decoded.gyroY, closeTo(0.0, 0.001));
    expect(decoded.gyroZ, closeTo(0.0, 0.001));
    expect(decoded.quatW, closeTo(0.0, 0.001));
    expect(decoded.quatX, closeTo(0.0, 0.001));
    expect(decoded.quatY, closeTo(0.0, 0.001));
    expect(decoded.quatZ, closeTo(0.0, 0.001));
  });

  test('IMUSample edge case: negative floats', () {
    final sample = IMUSample()
      ..relativeTimestampMs = Int64(500)
      ..accX = -9.81
      ..accY = -0.5
      ..accZ = -2.3
      ..gyroX = -3.14
      ..gyroY = -1.57
      ..gyroZ = -0.78
      ..quatW = -0.5
      ..quatX = -0.5
      ..quatY = -0.5
      ..quatZ = -0.5;

    final bytes = sample.writeToBuffer();
    final decoded = IMUSample()..mergeFromBuffer(bytes);

    expect(decoded.relativeTimestampMs.toInt(), equals(500));
    expect(decoded.accX, closeTo(-9.81, 0.001));
    expect(decoded.accY, closeTo(-0.5, 0.001));
    expect(decoded.accZ, closeTo(-2.3, 0.001));
    expect(decoded.gyroX, closeTo(-3.14, 0.001));
    expect(decoded.gyroY, closeTo(-1.57, 0.001));
    expect(decoded.gyroZ, closeTo(-0.78, 0.001));
    expect(decoded.quatW, closeTo(-0.5, 0.001));
    expect(decoded.quatX, closeTo(-0.5, 0.001));
    expect(decoded.quatY, closeTo(-0.5, 0.001));
    expect(decoded.quatZ, closeTo(-0.5, 0.001));
  });

  test('IMUSample edge case: large uint64 timestamp', () {
    final largeTimestamp = Int64.parseInt('18446744073709551615'); // max uint64
    final sample = IMUSample()
      ..relativeTimestampMs = largeTimestamp
      ..accX = 1.0
      ..accY = 2.0
      ..accZ = 3.0
      ..gyroX = 0.1
      ..gyroY = 0.2
      ..gyroZ = 0.3
      ..quatW = 1.0
      ..quatX = 0.0
      ..quatY = 0.0
      ..quatZ = 0.0;

    final bytes = sample.writeToBuffer();
    final decoded = IMUSample()..mergeFromBuffer(bytes);

    expect(decoded.relativeTimestampMs, equals(largeTimestamp));
    expect(decoded.accX, closeTo(1.0, 0.001));
  });
}

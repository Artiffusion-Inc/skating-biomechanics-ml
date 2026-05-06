import 'package:flutter_test/flutter_test.dart';
import 'package:fixnum/fixnum.dart';
import 'package:edgesense_capture/export/protobuf_gen/imu.pb.dart';

void main() {
  test('IMUSample roundtrip encode/decode', () {
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
    expect(decoded.quatW, closeTo(1.0, 0.001));
  });
}

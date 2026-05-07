import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import '../../lib/export/manifest_builder.dart';

void main() {
  group('ManifestBuilder', () {
    test('build produces valid JSON with expected fields', () {
      final t0 = DateTime(2024, 1, 1, 12, 0, 0);
      final manifest = ManifestBuilder.build(
        t0: t0,
        durationMs: 1500,
        videoFilename: 'capture_20240101_120000.mp4',
        videoWidth: 1920,
        videoHeight: 1080,
        videoFps: 60,
        leftImuFilename: 'capture_20240101_120000_left.pb',
        rightImuFilename: 'capture_20240101_120000_right.pb',
        leftRef: {
          'quat_ref': [1.0, 0.0, 0.0, 0.0],
          'calibrated_at': t0.toIso8601String(),
        },
        rightRef: {
          'quat_ref': [0.99, 0.01, 0.0, 0.0],
          'calibrated_at': t0.toIso8601String(),
        },
      );

      final json = jsonDecode(manifest) as Map<String, dynamic>;

      expect(json['version'], equals('1.0'));
      expect(json['created_at'], equals('2024-01-01T12:00:00.000'));
      expect(json['t0_ms'], equals(t0.millisecondsSinceEpoch));
      expect(json['duration_ms'], equals(1500));

      final video = json['video'] as Map<String, dynamic>;
      expect(video['filename'], equals('capture_20240101_120000.mp4'));
      expect(video['fps'], equals(60));
      expect(video['width'], equals(1920));
      expect(video['height'], equals(1080));
      expect(video['start_offset_ms'], equals(0));

      final imu = json['imu'] as Map<String, dynamic>;
      final left = imu['left'] as Map<String, dynamic>;
      expect(left['filename'], equals('capture_20240101_120000_left.pb'));
      expect(left['sample_rate_hz'], equals(100));

      final right = imu['right'] as Map<String, dynamic>;
      expect(right['filename'], equals('capture_20240101_120000_right.pb'));
      expect(right['sample_rate_hz'], equals(100));

      final calibration = json['calibration'] as Map<String, dynamic>;
      expect(calibration['left']['quat_ref'], equals([1.0, 0.0, 0.0, 0.0]));
      expect(calibration['right']['quat_ref'], equals([0.99, 0.01, 0.0, 0.0]));
    });
  });
}

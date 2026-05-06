import 'dart:convert';

class ManifestBuilder {
  static String build({
    required DateTime t0,
    required int durationMs,
    required String videoFilename,
    required int videoWidth,
    required int videoHeight,
    required int videoFps,
    required String leftImuFilename,
    required String rightImuFilename,
    required Map<String, dynamic> leftRef,
    required Map<String, dynamic> rightRef,
  }) {
    final manifest = {
      'version': '1.0',
      'created_at': t0.toIso8601String(),
      't0_ms': t0.millisecondsSinceEpoch,
      'duration_ms': durationMs,
      'video': {
        'filename': videoFilename,
        'fps': videoFps,
        'width': videoWidth,
        'height': videoHeight,
        'start_offset_ms': 0,
      },
      'imu': {
        'left': {
          'filename': leftImuFilename,
          'sample_rate_hz': 100,
          'start_offset_ms': 0,
        },
        'right': {
          'filename': rightImuFilename,
          'sample_rate_hz': 100,
          'start_offset_ms': 0,
        },
      },
      'calibration': {
        'left': leftRef,
        'right': rightRef,
      },
    };
    return const JsonEncoder.withIndent('  ').convert(manifest);
  }
}

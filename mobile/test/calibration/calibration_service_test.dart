import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/calibration/calibration_service.dart';

void main() {
  group('CalibrationService', () {
    late CalibrationService service;

    setUp(() {
      service = CalibrationService();
    });

    test('calibrate returns normalized quaternion', () {
      final quats = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
      ];
      final result = service.calibrate(quats);
      expect(result[0], closeTo(1.0, 0.001));
      expect(result[1], closeTo(0.0, 0.001));
      expect(result[2], closeTo(0.0, 0.001));
      expect(result[3], closeTo(0.0, 0.001));
    });

    test('calibrate averages multiple quaternions', () {
      final quats = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
      ];
      final result = service.calibrate(quats);
      // Average then normalize: [0.5, 0.5, 0, 0] → norm = sqrt(0.5) → [0.707, 0.707, 0, 0]
      expect(result[0], closeTo(0.707, 0.01));
      expect(result[1], closeTo(0.707, 0.01));
      expect(result[2], closeTo(0.0, 0.001));
      expect(result[3], closeTo(0.0, 0.001));
    });

    test('calibrate with near-zero values does not divide by zero', () {
      final quats = [
        [0.0, 0.0, 0.0, 0.0],
      ];
      final result = service.calibrate(quats);
      // Division by zero protection: scale = 1 when norm == 0
      expect(result, equals([0.0, 0.0, 0.0, 0.0]));
    });

    test('leftRef and rightRef are initially null', () {
      expect(service.leftRef, isNull);
      expect(service.rightRef, isNull);
    });
  });
}

import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/calibration/calibration_service.dart';

void main() {
  group('CalibrationService', () {
    test('computes average quaternion from identical inputs', () {
      final service = CalibrationService();
      final samples = List.generate(10, (_) => [1.0, 0.0, 0.0, 0.0]);
      final result = service.calibrate(samples);
      expect(result[0], closeTo(1.0, 0.001));
      expect(result[1], closeTo(0.0, 0.001));
      expect(result[2], closeTo(0.0, 0.001));
      expect(result[3], closeTo(0.0, 0.001));
    });

    test('normalizes the averaged quaternion', () {
      final service = CalibrationService();
      final samples = List.generate(10, (_) => [2.0, 2.0, 2.0, 2.0]);
      final result = service.calibrate(samples);
      // Norm should be 1.0
      final norm = result.fold(0.0, (s, v) => s + v * v);
      expect(norm, closeTo(1.0, 0.001));
    });

    test('throws on empty samples', () {
      final service = CalibrationService();
      expect(() => service.calibrate([]), throwsA(isA<AssertionError>()));
    });
  });
}

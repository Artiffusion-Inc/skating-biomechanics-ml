import 'dart:math' as math;

class CalibrationService {
  List<double>? leftRef;
  List<double>? rightRef;

  List<double> calibrate(List<List<double>> quaternions) {
    assert(quaternions.isNotEmpty);
    double sumW = 0, sumX = 0, sumY = 0, sumZ = 0;
    for (final q in quaternions) {
      sumW += q[0];
      sumX += q[1];
      sumY += q[2];
      sumZ += q[3];
    }
    final n = quaternions.length.toDouble();
    final mean = [sumW / n, sumX / n, sumY / n, sumZ / n];
    return _normalize(mean);
  }

  List<double> _normalize(List<double> q) {
    final norm = q.fold(0.0, (s, v) => s + v * v);
    final scale = 1.0 / (norm == 0 ? 1 : math.sqrt(norm));
    return [q[0] * scale, q[1] * scale, q[2] * scale, q[3] * scale];
  }
}

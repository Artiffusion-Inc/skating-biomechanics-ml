import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'capture_state.dart';
import '../ble/ble_manager.dart';
import '../ble/wt901_parser.dart';
import '../camera/recorder.dart';

class CaptureController extends ChangeNotifier {
  DateTime? t0;
  CaptureStatus status = CaptureStatus.idle;
  final List<Map<String, dynamic>> _leftBuffer = [];
  final List<Map<String, dynamic>> _rightBuffer = [];
  StreamSubscription? _streamSubscription;

  DateTime? get startTime => t0;

  Future<CaptureResult?> start({
    required BleManager bleManager,
    required CameraRecorder cameraRecorder,
    required void Function(double edgeAngle) onLeftEdgeAngle,
    required void Function(double edgeAngle) onRightEdgeAngle,
  }) async {
    if (status == CaptureStatus.recording) return null;

    _leftBuffer.clear();
    _rightBuffer.clear();

    status = CaptureStatus.initializing;
    notifyListeners();

    await bleManager.connectAll();
    await cameraRecorder.startRecording();

    final stopwatch = Stopwatch()..start();
    t0 = DateTime.now();
    status = CaptureStatus.recording;
    notifyListeners();

    _streamSubscription = bleManager.startStreams().listen((pair) {
      final relativeMs = stopwatch.elapsed.inMilliseconds;

      final left = pair.$1;
      if (left != null) {
        final sample = _toMap(left, relativeMs);
        _leftBuffer.add(sample);
        if (left.quatW != null) {
          final edgeAngle = _computeEdgeAngle(left);
          onLeftEdgeAngle(edgeAngle);
        }
      }

      final right = pair.$2;
      if (right != null) {
        final sample = _toMap(right, relativeMs);
        _rightBuffer.add(sample);
        if (right.quatW != null) {
          final edgeAngle = _computeEdgeAngle(right);
          onRightEdgeAngle(edgeAngle);
        }
      }
    });

    return null;
  }

  Map<String, dynamic> _toMap(WT901Packet packet, int relativeMs) {
    return {
      'relative_timestamp_ms': relativeMs,
      'acc_x': packet.accX,
      'acc_y': packet.accY,
      'acc_z': packet.accZ,
      'gyro_x': packet.gyroX,
      'gyro_y': packet.gyroY,
      'gyro_z': packet.gyroZ,
      'quat_w': packet.quatW,
      'quat_x': packet.quatX,
      'quat_y': packet.quatY,
      'quat_z': packet.quatZ,
    };
  }

  double _computeEdgeAngle(WT901Packet p) {
    final qx = p.quatX ?? 0;
    final qy = p.quatY ?? 0;
    final qz = p.quatZ ?? 0;
    final qw = p.quatW ?? 0;
    final roll = math.atan2(
      2.0 * (qw * qx + qy * qz),
      1.0 - 2.0 * (qx * qx + qy * qy),
    );
    return 180.0 / math.pi * roll;
  }

  Future<CaptureResult> stop({
    required BleManager bleManager,
    required CameraRecorder cameraRecorder,
  }) async {
    status = CaptureStatus.stopping;
    notifyListeners();

    await _streamSubscription?.cancel();
    final videoFile = await cameraRecorder.stopRecording();
    await bleManager.disconnectAll();

    status = CaptureStatus.idle;
    notifyListeners();

    return CaptureResult(
      videoPath: videoFile.path,
      leftSamples: List.from(_leftBuffer),
      rightSamples: List.from(_rightBuffer),
      t0: t0!,
    );
  }
}

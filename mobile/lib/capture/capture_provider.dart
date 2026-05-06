import 'package:flutter/foundation.dart';

import '../ble/ble_manager.dart';
import '../camera/recorder.dart';
import 'capture_repository.dart';
import 'capture_state.dart';

class CaptureProvider extends ChangeNotifier {
  final CaptureRepository _repo;

  CaptureStatus status = CaptureStatus.idle;
  DateTime? get startTime => _repo.startTime;
  int get leftSampleCount => _repo.leftSampleCount;
  int get rightSampleCount => _repo.rightSampleCount;

  CaptureProvider({required BleManager bleManager, required CameraRecorder cameraRecorder})
      : _repo = CaptureRepository(bleManager: bleManager, cameraRecorder: cameraRecorder);

  Future<CaptureResult?> start({
    required void Function(double edgeAngle) onLeftEdgeAngle,
    required void Function(double edgeAngle) onRightEdgeAngle,
  }) async {
    if (status == CaptureStatus.recording) return null;

    status = CaptureStatus.initializing;
    notifyListeners();

    final result = await _repo.start(
      onLeftEdgeAngle: onLeftEdgeAngle,
      onRightEdgeAngle: onRightEdgeAngle,
    );

    if (result == null) {
      status = CaptureStatus.recording;
      notifyListeners();
    } else {
      status = CaptureStatus.error;
      notifyListeners();
    }

    return result;
  }

  Future<CaptureResult> stop() async {
    status = CaptureStatus.stopping;
    notifyListeners();

    final result = await _repo.stop();

    status = CaptureStatus.idle;
    notifyListeners();

    return result;
  }

  @override
  void dispose() {
    _repo.dispose();
    super.dispose();
  }
}

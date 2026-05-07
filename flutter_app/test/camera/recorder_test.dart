import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/camera/recorder.dart';

void main() {
  group('CameraRecorder', () {
    test('initializes with empty cameras', () async {
      final recorder = CameraRecorder();
      await recorder.initialize([]);
      expect(recorder.isInitialized, isFalse);
    });

    test('throws when starting without initialization', () {
      final recorder = CameraRecorder();
      expect(() => recorder.startRecording(), throwsStateError);
    });

    test('throws when stopping without initialization', () {
      final recorder = CameraRecorder();
      expect(() => recorder.stopRecording(), throwsStateError);
    });

    test('dispose sets controller to null', () async {
      final recorder = CameraRecorder();
      await recorder.initialize([]);
      recorder.dispose();
      expect(recorder.controller, isNull);
    });
  });
}

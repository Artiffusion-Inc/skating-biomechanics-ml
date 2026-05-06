import 'package:camera/camera.dart';

class CameraRecorder {
  CameraController? _controller;
  bool get isInitialized => _controller?.value.isInitialized ?? false;

  Future<void> initialize(List<CameraDescription> cameras) async {
    if (cameras.isEmpty) {
      _controller = null;
      return;
    }
    _controller = CameraController(
      cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      ),
      ResolutionPreset.high,
      enableAudio: false,
      fps: 60,
    );
    await _controller!.initialize();
  }

  Future<void> startRecording() async {
    if (_controller == null || !isInitialized) {
      throw StateError('Camera not initialized');
    }
    await _controller!.startVideoRecording();
  }

  Future<XFile> stopRecording() async {
    if (_controller == null || !isInitialized) {
      throw StateError('Camera not initialized');
    }
    return await _controller!.stopVideoRecording();
  }

  CameraController? get controller => _controller;

  Future<void> dispose() async {
    await _controller?.dispose();
    _controller = null;
  }
}

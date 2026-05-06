import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class CameraRecorder extends ChangeNotifier {
  CameraController? _controller;
  CameraLensDirection _lensDirection = CameraLensDirection.back;
  ResolutionPreset _resolution = ResolutionPreset.high;
  bool _showGrid = false;
  bool _orientationLocked = false;

  bool get isInitialized => _controller?.value.isInitialized ?? false;
  bool get showGrid => _showGrid;
  bool get orientationLocked => _orientationLocked;
  ResolutionPreset get resolution => _resolution;
  CameraController? get controller => _controller;

  Future<void> initialize(List<CameraDescription> cameras) async {
    if (cameras.isEmpty) {
      _controller = null;
      return;
    }
    _controller?.dispose();
    _controller = CameraController(
      cameras.firstWhere(
        (c) => c.lensDirection == _lensDirection,
        orElse: () => cameras.first,
      ),
      _resolution,
      enableAudio: false,
      fps: 60,
    );
    await _controller!.initialize();
    if (_orientationLocked) {
      await SystemChrome.setPreferredOrientations([
        DeviceOrientation.portraitUp,
      ]);
    }
    notifyListeners();
  }

  Future<void> toggleCamera() async {
    _lensDirection = _lensDirection == CameraLensDirection.back
        ? CameraLensDirection.front
        : CameraLensDirection.back;
    final cameras = await availableCameras();
    await initialize(cameras);
  }

  Future<void> setResolution(ResolutionPreset r) async {
    _resolution = r;
    final cameras = await availableCameras();
    await initialize(cameras);
  }

  Future<void> setOrientationLocked(bool locked) async {
    _orientationLocked = locked;
    if (locked) {
      await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    } else {
      await SystemChrome.setPreferredOrientations([]);
    }
    notifyListeners();
  }

  void toggleGrid() {
    _showGrid = !_showGrid;
    notifyListeners();
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

  @override
  void dispose() {
    _controller?.dispose();
    _controller = null;
    super.dispose();
  }
}

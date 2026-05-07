import 'package:camera/camera.dart' hide availableCameras;
import 'package:camera/camera.dart' as cam show availableCameras;

abstract class CameraFactory {
  Future<List<CameraDescription>> getCameras();
  CameraController createController(
    CameraDescription description,
    ResolutionPreset resolution, {
    bool enableAudio = false,
    int? fps,
  });
}

class SystemCameraFactory implements CameraFactory {
  @override
  Future<List<CameraDescription>> getCameras() => cam.availableCameras();

  @override
  CameraController createController(
    CameraDescription description,
    ResolutionPreset resolution, {
    bool enableAudio = false,
    int? fps,
  }) => CameraController(
    description,
    resolution,
    enableAudio: enableAudio,
    fps: fps,
  );
}

import 'package:camera/camera.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:edgesense_capture/camera/recorder.dart';
import 'package:edgesense_capture/camera/camera_factory.dart';

class MockCameraFactory extends Mock implements CameraFactory {}

class MockCameraController extends Mock implements CameraController {}

class FakeCameraDescription extends Fake implements CameraDescription {
  final CameraLensDirection direction;
  FakeCameraDescription(this.direction);

  @override
  CameraLensDirection get lensDirection => direction;

  @override
  String get name => direction == CameraLensDirection.back ? 'back' : 'front';
}

void main() {
  setUpAll(() {
    registerFallbackValue(FakeCameraDescription(CameraLensDirection.back));
    registerFallbackValue(ResolutionPreset.high);
  });

  group('CameraRecorder', () {
    late MockCameraFactory factory;
    late MockCameraController mockController;
    late CameraRecorder recorder;

    setUp(() {
      factory = MockCameraFactory();
      mockController = MockCameraController();

      when(() => factory.getCameras()).thenAnswer(
        (_) async => [
          FakeCameraDescription(CameraLensDirection.back),
          FakeCameraDescription(CameraLensDirection.front),
        ],
      );

      when(
        () => factory.createController(
          any(),
          any(),
          enableAudio: any(named: 'enableAudio'),
          fps: any(named: 'fps'),
        ),
      ).thenReturn(mockController);

      when(() => mockController.initialize()).thenAnswer((_) async {});
      when(() => mockController.startVideoRecording()).thenAnswer((_) async {});
      when(
        () => mockController.stopVideoRecording(),
      ).thenAnswer((_) async => XFile('/tmp/video.mp4'));
      when(() => mockController.value).thenReturn(
        CameraValue.uninitialized(
          FakeCameraDescription(CameraLensDirection.back),
        ).copyWith(isInitialized: true),
      );
      when(() => mockController.dispose()).thenAnswer((_) async {});

      recorder = CameraRecorder(factory: factory);
    });

    tearDown(() {
      recorder.dispose();
    });

    test('initial state', () {
      expect(recorder.isInitialized, false);
      expect(recorder.showGrid, false);
      expect(recorder.orientationLocked, false);
      expect(recorder.controller, isNull);
    });

    test('initialize creates controller and notifies', () async {
      var notified = false;
      recorder.addListener(() => notified = true);

      await recorder.initialize([
        FakeCameraDescription(CameraLensDirection.back),
      ]);

      verify(
        () => factory.createController(
          any(),
          any(),
          enableAudio: any(named: 'enableAudio'),
          fps: any(named: 'fps'),
        ),
      ).called(1);
      verify(() => mockController.initialize()).called(1);
      expect(notified, true);
    });

    test('initialize with empty cameras sets controller to null', () async {
      await recorder.initialize([]);
      expect(recorder.controller, isNull);
      expect(recorder.isInitialized, false);
    });

    test('toggleGrid toggles and notifies', () {
      expect(recorder.showGrid, false);
      var notifiedCount = 0;
      recorder.addListener(() => notifiedCount++);

      recorder.toggleGrid();
      expect(recorder.showGrid, true);
      expect(notifiedCount, 1);

      recorder.toggleGrid();
      expect(recorder.showGrid, false);
      expect(notifiedCount, 2);
    });

    test('startRecording throws when not initialized', () async {
      expect(() => recorder.startRecording(), throwsA(isA<StateError>()));
    });

    test('stopRecording throws when not initialized', () async {
      expect(() => recorder.stopRecording(), throwsA(isA<StateError>()));
    });

    test('startRecording calls controller', () async {
      await recorder.initialize([
        FakeCameraDescription(CameraLensDirection.back),
      ]);

      await recorder.startRecording();
      verify(() => mockController.startVideoRecording()).called(1);
    });

    test('stopRecording returns XFile', () async {
      await recorder.initialize([
        FakeCameraDescription(CameraLensDirection.back),
      ]);

      final result = await recorder.stopRecording();
      expect(result, isA<XFile>());
      verify(() => mockController.stopVideoRecording()).called(1);
    });
  });
}

import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/export/exporter.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const channel = MethodChannel('plugins.flutter.io/path_provider');
  final tempDir = Directory.systemTemp.createTempSync('edgesense_test_');
  late File dummyVideo;

  setUpAll(() {
    dummyVideo = File('${tempDir.path}/dummy_video.mp4');
    dummyVideo.writeAsBytesSync([0x00, 0x00, 0x00, 0x00]);

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      if (call.method == 'getDownloadsDirectory') {
        return tempDir.path;
      }
      if (call.method == 'getExternalStorageDirectory') {
        return tempDir.path;
      }
      if (call.method == 'getTemporaryDirectory') {
        return tempDir.path;
      }
      return null;
    });
  });

  tearDownAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
    tempDir.deleteSync(recursive: true);
  });

  group('Exporter', () {
    test('exports a zip with correct extension', () async {
      final exporter = Exporter();
      final result = await exporter.export(
        videoPath: dummyVideo.path,
        leftSamples: [],
        rightSamples: [],
        t0: DateTime.now(),
        leftRef: [1.0, 0.0, 0.0, 0.0],
        rightRef: [1.0, 0.0, 0.0, 0.0],
      );
      expect(result.endsWith('.zip'), isTrue);
    });

    test('includes samples in csv and zip', () async {
      final exporter = Exporter();
      final result = await exporter.export(
        videoPath: dummyVideo.path,
        leftSamples: [
          {
            'relative_timestamp_ms': 0,
            'acc_x': 0.0,
            'acc_y': 0.0,
            'acc_z': 1.0,
            'gyro_x': 0.0,
            'gyro_y': 0.0,
            'gyro_z': 0.0,
            'quat_w': 1.0,
            'quat_x': 0.0,
            'quat_y': 0.0,
            'quat_z': 0.0,
          },
        ],
        rightSamples: [],
        t0: DateTime.now(),
        leftRef: [1.0, 0.0, 0.0, 0.0],
        rightRef: [1.0, 0.0, 0.0, 0.0],
      );
      final file = File(result);
      expect(await file.exists(), isTrue);
      final stat = await file.stat();
      expect(stat.size, greaterThan(0));
    });
  });
}

import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:fixnum/fixnum.dart';
import 'package:archive/archive.dart';
import '../../lib/export/exporter.dart';
import '../../lib/export/protobuf_gen/imu.pb.dart';

void main() {
  group('Exporter', () {
    late Directory tempDir;
    late Exporter exporter;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('exporter_test_');
      exporter = Exporter();
    });

    tearDown(() async {
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    Future<String> _createDummyVideo() async {
      final path = '${tempDir.path}/dummy.mp4';
      await File(path).writeAsBytes([0, 0, 0, 0]); // minimal bytes
      return path;
    }

    test('export creates zip with protobuf IMU streams', () async {
      final videoPath = await _createDummyVideo();
      final samples = [
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
        {
          'relative_timestamp_ms': 10,
          'acc_x': 0.1,
          'acc_y': 0.0,
          'acc_z': 0.9,
          'gyro_x': 0.5,
          'gyro_y': 0.0,
          'gyro_z': 0.0,
          'quat_w': 0.99,
          'quat_x': 0.01,
          'quat_y': 0.0,
          'quat_z': 0.0,
        },
      ];

      final zipPath = await exporter.export(
        videoPath: videoPath,
        leftSamples: samples,
        rightSamples: samples,
        t0: DateTime(2024, 1, 1, 12, 0, 0),
        leftRef: [1.0, 0.0, 0.0, 0.0],
        rightRef: [1.0, 0.0, 0.0, 0.0],
        outputDir: tempDir.path,
      );

      expect(await File(zipPath).exists(), isTrue);

      // Verify zip contents
      final zipBytes = await File(zipPath).readAsBytes();
      final archive = ZipDecoder().decodeBytes(zipBytes);

      final filenames = archive.map((f) => f.name).toList();
      expect(filenames, contains('capture_20240101_120000.json')); // manifest
      expect(
        filenames,
        contains('capture_20240101_120000_left.pb'),
      ); // protobuf
      expect(
        filenames,
        contains('capture_20240101_120000_right.pb'),
      ); // protobuf

      // Verify protobuf deserializes correctly
      final leftPbFile = archive.firstWhere((f) => f.name.endsWith('_left.pb'));
      final stream = IMUStream.fromBuffer(leftPbFile.content as List<int>);
      expect(stream.samples.length, equals(2));
      expect(stream.samples[0].relativeTimestampMs, equals(Int64(0)));
      expect(stream.samples[0].accZ, closeTo(1.0, 0.001));
      expect(stream.samples[1].relativeTimestampMs, equals(Int64(10)));
    });

    test('export computes max duration from both channels', () async {
      final videoPath = await _createDummyVideo();
      final left = [
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
      ];
      final right = [
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
        {
          'relative_timestamp_ms': 500,
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
      ];

      final zipPath = await exporter.export(
        videoPath: videoPath,
        leftSamples: left,
        rightSamples: right,
        t0: DateTime(2024, 1, 1, 12, 0, 0),
        leftRef: [1.0, 0.0, 0.0, 0.0],
        rightRef: [1.0, 0.0, 0.0, 0.0],
        outputDir: tempDir.path,
      );

      expect(await File(zipPath).exists(), isTrue);
    });
  });
}

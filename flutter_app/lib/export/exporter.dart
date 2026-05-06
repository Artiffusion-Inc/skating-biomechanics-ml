import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
import 'package:fixnum/fixnum.dart' as fixnum;
import 'package:path_provider/path_provider.dart';
import 'manifest_builder.dart';
import 'protobuf_gen/imu.pb.dart';

class Exporter {
  Future<String> export({
    required String videoPath,
    required List<Map<String, dynamic>> leftSamples,
    required List<Map<String, dynamic>> rightSamples,
    required DateTime t0,
    required List<double> leftRef,
    required List<double> rightRef,
    int videoWidth = 1920,
    int videoHeight = 1080,
    int videoFps = 60,
    String? outputDir,
  }) async {
    final archive = Archive();

    final manifest = ManifestBuilder.build(
      t0: t0,
      durationMs: _computeDuration(leftSamples, rightSamples),
      videoFilename: 'video.mp4',
      videoWidth: videoWidth,
      videoHeight: videoHeight,
      videoFps: videoFps,
      leftImuFilename: 'left.imu',
      rightImuFilename: 'right.imu',
      leftRef: {'quat_ref': leftRef, 'calibrated_at': t0.toIso8601String()},
      rightRef: {'quat_ref': rightRef, 'calibrated_at': t0.toIso8601String()},
    );
    final manifestBytes = Uint8List.fromList(manifest.codeUnits);
    archive.addFile(ArchiveFile('manifest.json', manifestBytes.length, manifestBytes));

    final videoFile = File(videoPath);
    if (await videoFile.exists()) {
      final videoBytes = await videoFile.readAsBytes();
      archive.addFile(ArchiveFile('video.mp4', videoBytes.length, videoBytes));
    }

    final leftProto = _buildProtobuf(leftSamples);
    archive.addFile(ArchiveFile('left.imu', leftProto.length, leftProto));

    final rightProto = _buildProtobuf(rightSamples);
    archive.addFile(ArchiveFile('right.imu', rightProto.length, rightProto));

    final zipEncoder = ZipEncoder();
    final zipBytes = zipEncoder.encode(archive);
    final dir = outputDir != null ? Directory(outputDir) : await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final zipPath = '${dir.path}/capture_$timestamp.esense.zip';
    await File(zipPath).writeAsBytes(zipBytes!);
    return zipPath;
  }

  int _computeDuration(List<Map<String, dynamic>> left, List<Map<String, dynamic>> right) {
    int maxTs = 0;
    for (final s in left) {
      final ts = s['relative_timestamp_ms'] as int? ?? 0;
      if (ts > maxTs) maxTs = ts;
    }
    for (final s in right) {
      final ts = s['relative_timestamp_ms'] as int? ?? 0;
      if (ts > maxTs) maxTs = ts;
    }
    return maxTs;
  }

  Uint8List _buildProtobuf(List<Map<String, dynamic>> samples) {
    final stream = IMUStream(
      samples: samples.map((s) => IMUSample(
        relativeTimestampMs: fixnum.Int64(s['relative_timestamp_ms'] as int),
        accX: (s['acc_x'] as double?) ?? 0,
        accY: (s['acc_y'] as double?) ?? 0,
        accZ: (s['acc_z'] as double?) ?? 0,
        gyroX: (s['gyro_x'] as double?) ?? 0,
        gyroY: (s['gyro_y'] as double?) ?? 0,
        gyroZ: (s['gyro_z'] as double?) ?? 0,
        quatW: (s['quat_w'] as double?) ?? 0,
        quatX: (s['quat_x'] as double?) ?? 0,
        quatY: (s['quat_y'] as double?) ?? 0,
        quatZ: (s['quat_z'] as double?) ?? 0,
      )).toList(),
    );
    return stream.writeToBuffer();
  }
}

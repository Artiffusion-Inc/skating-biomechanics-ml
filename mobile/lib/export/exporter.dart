import 'dart:io';
import 'package:archive/archive.dart';
import 'package:path_provider/path_provider.dart';
import 'manifest_builder.dart';

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
    final dir = await _ensureExportDir(outputDir);
    final basename = _formatBasename(t0);

    final videoFile = File(videoPath);
    final targetVideo = File('${dir.path}/$basename.mp4');
    if (await videoFile.exists()) {
      await videoFile.copy(targetVideo.path);
    }

    final leftCsvPath = '${dir.path}/${basename}_left.csv';
    await _writeCsv(leftCsvPath, leftSamples);

    final rightCsvPath = '${dir.path}/${basename}_right.csv';
    await _writeCsv(rightCsvPath, rightSamples);

    final manifest = ManifestBuilder.build(
      t0: t0,
      durationMs: _computeDuration(leftSamples, rightSamples),
      videoFilename: '$basename.mp4',
      videoWidth: videoWidth,
      videoHeight: videoHeight,
      videoFps: videoFps,
      leftImuFilename: '${basename}_left.csv',
      rightImuFilename: '${basename}_right.csv',
      leftRef: {'quat_ref': leftRef, 'calibrated_at': t0.toIso8601String()},
      rightRef: {'quat_ref': rightRef, 'calibrated_at': t0.toIso8601String()},
    );
    final manifestPath = '${dir.path}/$basename.json';
    await File(manifestPath).writeAsString(manifest);

    final archive = Archive();
    if (await targetVideo.exists()) {
      archive.addFile(await _archiveFile(targetVideo));
    }
    archive.addFile(await _archiveFile(File(leftCsvPath)));
    archive.addFile(await _archiveFile(File(rightCsvPath)));
    archive.addFile(await _archiveFile(File(manifestPath)));

    final zipEncoder = ZipEncoder();
    final zipBytes = zipEncoder.encode(archive);
    final zipPath = '${dir.path}/$basename.zip';
    await File(zipPath).writeAsBytes(zipBytes);
    return zipPath;
  }

  Future<Directory> _ensureExportDir(String? outputDir) async {
    if (outputDir != null) return Directory(outputDir);
    Directory? downloads;
    try {
      downloads = await getDownloadsDirectory();
    } catch (_) {}
    if (downloads == null) {
      try {
        downloads = await getExternalStorageDirectory();
      } catch (_) {}
    }
    downloads ??= await getTemporaryDirectory();
    final dir = Directory('${downloads.path}/EdgeSense');
    await dir.create(recursive: true);
    return dir;
  }

  String _formatBasename(DateTime dt) {
    final y = dt.year;
    final m = dt.month.toString().padLeft(2, '0');
    final d = dt.day.toString().padLeft(2, '0');
    final h = dt.hour.toString().padLeft(2, '0');
    final min = dt.minute.toString().padLeft(2, '0');
    final s = dt.second.toString().padLeft(2, '0');
    return 'capture_$y$m$d\_$h$min$s';
  }

  Future<void> _writeCsv(String path, List<Map<String, dynamic>> samples) async {
    final buffer = StringBuffer();
    buffer.writeln(
        'relative_timestamp_ms,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,quat_w,quat_x,quat_y,quat_z');
    for (final s in samples) {
      buffer.writeln(
          '${s['relative_timestamp_ms']},${s['acc_x']},${s['acc_y']},${s['acc_z']},${s['gyro_x']},${s['gyro_y']},${s['gyro_z']},${s['quat_w']},${s['quat_x']},${s['quat_y']},${s['quat_z']}');
    }
    await File(path).writeAsString(buffer.toString());
  }

  Future<ArchiveFile> _archiveFile(File file) async {
    final bytes = await file.readAsBytes();
    final name = file.path.split('/').last;
    return ArchiveFile(name, bytes.length, bytes);
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
}

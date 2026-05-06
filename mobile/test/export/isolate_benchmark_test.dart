import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:archive/archive.dart';

void main() {
  group('Isolate zip encoding benchmark', () {
    late List<MapEntry<String, List<int>>> entries;

    setUp(() {
      // Simulate a realistic export payload: ~2MB of files
      entries = [
        MapEntry(
          'video.mp4',
          Uint8List.fromList(List.generate(1024 * 1024, (i) => i % 256)),
        ),
        MapEntry(
          'left.pb',
          Uint8List.fromList(List.generate(500 * 1024, (i) => i % 256)),
        ),
        MapEntry(
          'right.pb',
          Uint8List.fromList(List.generate(500 * 1024, (i) => i % 256)),
        ),
        MapEntry(
          'manifest.json',
          Uint8List.fromList(List.generate(1024, (i) => i % 256)),
        ),
      ];
    });

    test('Isolate.run vs sync zip encoding', () async {
      // Warm up
      _buildZipSync(entries);
      await Isolate.run(() => _buildZipSync(entries));

      // Sync benchmark
      final swSync = Stopwatch()..start();
      for (var i = 0; i < 5; i++) {
        _buildZipSync(entries);
      }
      swSync.stop();

      // Isolate benchmark
      final swIso = Stopwatch()..start();
      for (var i = 0; i < 5; i++) {
        await Isolate.run(() => _buildZipSync(entries));
      }
      swIso.stop();

      final syncMs = swSync.elapsed.inMilliseconds / 5;
      final isoMs = swIso.elapsed.inMilliseconds / 5;

      // Isolate should not be dramatically slower (within 3x is acceptable)
      expect(
        isoMs,
        lessThan(syncMs * 3.0),
        reason: 'Isolate overhead too high: sync=${syncMs}ms, iso=${isoMs}ms',
      );
    });
  });
}

List<int> _buildZipSync(List<MapEntry<String, List<int>>> entries) {
  final archive = Archive();
  for (final e in entries) {
    archive.addFile(ArchiveFile(e.key, e.value.length, e.value));
  }
  return ZipEncoder().encode(archive);
}

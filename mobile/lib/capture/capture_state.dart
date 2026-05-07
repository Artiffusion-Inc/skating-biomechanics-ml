enum CaptureStatus { idle, initializing, recording, stopping, error }

class CaptureResult {
  final String videoPath;
  final List<Map<String, dynamic>> leftSamples;
  final List<Map<String, dynamic>> rightSamples;
  final DateTime t0;

  CaptureResult({
    required this.videoPath,
    required this.leftSamples,
    required this.rightSamples,
    required this.t0,
  });
}

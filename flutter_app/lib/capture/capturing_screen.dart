import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../ble/ble_manager.dart';
import '../capture/capture_controller.dart';
import '../capture/capture_state.dart';
import '../calibration/calibration_service.dart';
import '../camera/recorder.dart';
import '../export/exporter.dart';
import '../overlay/edge_overlay.dart';

class CapturingScreen extends StatefulWidget {
  final void Function(String? exportPath) onComplete;
  const CapturingScreen({super.key, required this.onComplete});

  @override
  State<CapturingScreen> createState() => _CapturingScreenState();
}

class _CapturingScreenState extends State<CapturingScreen> {
  double _leftAngle = 0;
  double _rightAngle = 0;
  Duration _elapsed = Duration.zero;
  bool _stopping = false;
  Timer? _elapsedTimer;

  @override
  void initState() {
    super.initState();
    _startCapture();
  }

  @override
  void dispose() {
    _elapsedTimer?.cancel();
    super.dispose();
  }

  Future<void> _startCapture() async {
    final bleManager = context.read<BleManager>();
    final cameraRecorder = context.read<CameraRecorder>();
    final captureController = context.read<CaptureController>();

    try {
      await captureController.start(
        bleManager: bleManager,
        cameraRecorder: cameraRecorder,
        onLeftEdgeAngle: (a) => setState(() => _leftAngle = a),
        onRightEdgeAngle: (a) => setState(() => _rightAngle = a),
      );

      final start = captureController.startTime;
      if (start != null) {
        _elapsedTimer = Timer.periodic(const Duration(milliseconds: 200), (_) {
          if (mounted && captureController.status == CaptureStatus.recording) {
            setState(() => _elapsed = DateTime.now().difference(start));
          }
        });
      }
    } catch (e) {
      if (mounted) widget.onComplete(null);
    }
  }

  Future<void> _stopCapture() async {
    if (_stopping) return;
    setState(() => _stopping = true);
    _elapsedTimer?.cancel();

    final bleManager = context.read<BleManager>();
    final cameraRecorder = context.read<CameraRecorder>();
    final captureController = context.read<CaptureController>();
    final calibration = context.read<CalibrationService>();

    try {
      final result = await captureController.stop(
        bleManager: bleManager,
        cameraRecorder: cameraRecorder,
      );

      final leftRef = calibration.leftRef ?? [1.0, 0.0, 0.0, 0.0];
      final rightRef = calibration.rightRef ?? [1.0, 0.0, 0.0, 0.0];

      final exportPath = await Exporter().export(
        videoPath: result.videoPath,
        leftSamples: result.leftSamples,
        rightSamples: result.rightSamples,
        t0: result.t0,
        leftRef: leftRef,
        rightRef: rightRef,
      );
      if (mounted) widget.onComplete(exportPath);
    } catch (e) {
      if (mounted) widget.onComplete(null);
    }
  }

  @override
  Widget build(BuildContext context) {
    final recorder = context.read<CameraRecorder>();

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (recorder.isInitialized && recorder.controller != null)
            Center(
              child: AspectRatio(
                aspectRatio: recorder.controller!.value.aspectRatio,
                child: CameraPreview(recorder.controller!),
              ),
            ),
          EdgeOverlay(leftAngle: _leftAngle, rightAngle: _rightAngle),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.red.shade900,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.fiber_manual_record, color: Colors.red.shade300, size: 16),
                        const SizedBox(width: 6),
                        Text(
                          'REC  ${_elapsed.inMinutes.toString().padLeft(2, '0')}:${(_elapsed.inSeconds % 60).toString().padLeft(2, '0')}',
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                  ),
                  Consumer<CaptureController>(
                    builder: (ctx, ctrl, _) => Text(
                      'L:${ctrl.leftSampleCount}  R:${ctrl.rightSampleCount}',
                      style: const TextStyle(fontSize: 12, color: Colors.white70),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _stopping ? null : _stopCapture,
        icon: _stopping
            ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
            : const Icon(Icons.stop),
        label: Text(_stopping ? 'Сохранение...' : 'Стоп'),
        backgroundColor: Colors.red,
      ),
    );
  }
}
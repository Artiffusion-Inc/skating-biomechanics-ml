import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'providers/app_providers.dart';
import 'theme/app_theme.dart';
import 'ble/ble_manager.dart';
import 'camera/recorder.dart';
import 'capture/capture_controller.dart';
import 'capture/capture_state.dart';
import 'overlay/edge_overlay.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AppProviders(child: EdgeSenseApp()));
}

class EdgeSenseApp extends StatelessWidget {
  const EdgeSenseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EdgeSense Capture',
      theme: AppTheme.dark,
      home: const CaptureScreen(),
    );
  }
}

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  double _leftAngle = 0;
  double _rightAngle = 0;

  @override
  Widget build(BuildContext context) {
    final captureController = context.watch<CaptureController>();
    final status = captureController.status;

    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeSense Capture'),
        actions: [
          if (status == CaptureStatus.recording)
            const Padding(
              padding: EdgeInsets.only(right: 16),
              child: Row(
                children: [
                  Icon(Icons.fiber_manual_record, color: Colors.red),
                  SizedBox(width: 4),
                  Text('REC', style: TextStyle(color: Colors.red)),
                ],
              ),
            ),
        ],
      ),
      body: Stack(
        children: [
          const Center(child: Text('Camera preview placeholder')),
          EdgeOverlay(leftAngle: _leftAngle, rightAngle: _rightAngle),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          if (status == CaptureStatus.idle) {
            _startCapture();
          } else if (status == CaptureStatus.recording) {
            _stopCapture();
          }
        },
        label: Text(status == CaptureStatus.recording ? 'Stop' : 'Start'),
        icon: Icon(status == CaptureStatus.recording ? Icons.stop : Icons.videocam),
      ),
    );
  }

  void _startCapture() {
    final bleManager = context.read<BleManager>();
    final cameraRecorder = context.read<CameraRecorder>();
    final captureController = context.read<CaptureController>();

    captureController.start(
      bleManager: bleManager,
      cameraRecorder: cameraRecorder,
      onLeftEdgeAngle: (angle) => setState(() => _leftAngle = angle),
      onRightEdgeAngle: (angle) => setState(() => _rightAngle = angle),
    );
  }

  void _stopCapture() {
    final bleManager = context.read<BleManager>();
    final cameraRecorder = context.read<CameraRecorder>();
    final captureController = context.read<CaptureController>();

    captureController.stop(
      bleManager: bleManager,
      cameraRecorder: cameraRecorder,
    );
  }
}

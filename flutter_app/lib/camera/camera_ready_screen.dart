import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../ble/ble_manager.dart';
import '../metrics/metrics_screen.dart';
import 'recorder.dart';

class CameraReadyScreen extends StatefulWidget {
  final VoidCallback onStartCapture;
  const CameraReadyScreen({super.key, required this.onStartCapture});

  @override
  State<CameraReadyScreen> createState() => _CameraReadyScreenState();
}

class _CameraReadyScreenState extends State<CameraReadyScreen> {
  bool _initializing = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final recorder = context.read<CameraRecorder>();
      await recorder.initialize(cameras);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _initializing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final recorder = context.watch<CameraRecorder>();

    if (_initializing) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    if (_error != null || !recorder.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('Камера')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.videocam_off, size: 64, color: Colors.white54),
              const SizedBox(height: 16),
              Text(_error ?? 'Камера недоступна', style: const TextStyle(fontSize: 16)),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: _initCamera,
                icon: const Icon(Icons.refresh),
                label: const Text('Повторить'),
              ),
            ],
          ),
        ),
      );
    }

    final controller = recorder.controller!;
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: controller.value.aspectRatio,
              child: CameraPreview(controller),
            ),
          ),
          SafeArea(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              color: Colors.black54,
              child: Row(
                children: [
                  const Icon(Icons.bluetooth_connected, color: Colors.blue, size: 20),
                  const SizedBox(width: 8),
                  Consumer<BleManager>(
                    builder: (ctx, ble, _) => Text(
                      '${ble.leftDevice != null ? "L" : "—"} / ${ble.rightDevice != null ? "R" : "—"}',
                      style: const TextStyle(fontSize: 14),
                    ),
                  ),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.show_chart, color: Colors.white, size: 20),
                    tooltip: 'Датчики',
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const MetricsScreen()),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: widget.onStartCapture,
        icon: const Icon(Icons.fiber_manual_record),
        label: const Text('Запись'),
        backgroundColor: Colors.red,
      ),
    );
  }
}
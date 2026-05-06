import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'providers/app_providers.dart';
import 'theme/app_theme.dart';
import 'ble/ble_manager.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'camera/recorder.dart';
import 'capture/capture_controller.dart';
import 'capture/capture_state.dart';
import 'calibration/calibration_service.dart';
import 'export/exporter.dart';
import 'overlay/edge_overlay.dart';
import 'permissions/permission_service.dart';

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
      home: const AppShell(),
    );
  }
}

enum AppStep { permissions, bleScan, camera, capturing, exporting }

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  AppStep _step = AppStep.permissions;
  String? _exportPath;

  @override
  void initState() {
    super.initState();
    _checkPermissions();
  }

  Future<void> _checkPermissions() async {
    final granted = await PermissionService().requestAll();
    if (mounted) {
      setState(() => _step = granted ? AppStep.bleScan : AppStep.permissions);
    }
  }

  @override
  Widget build(BuildContext context) {
    return switch (_step) {
      AppStep.permissions => _PermissionsView(onGranted: () => setState(() => _step = AppStep.bleScan)),
      AppStep.bleScan => _BleScanView(
          onReady: () => setState(() => _step = AppStep.camera),
        ),
      AppStep.camera => _CameraReadyView(
          onStartCapture: () => setState(() => _step = AppStep.capturing),
        ),
      AppStep.capturing => _CapturingView(
          onStop: (path) => setState(() {
                _exportPath = path;
                _step = AppStep.exporting;
              }),
        ),
      AppStep.exporting => _ExportResultView(
          exportPath: _exportPath!,
          onNewCapture: () => setState(() => _step = AppStep.bleScan),
        ),
    };
  }
}

// --- Permissions ---

class _PermissionsView extends StatelessWidget {
  final VoidCallback onGranted;
  const _PermissionsView({required this.onGranted});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('EdgeSense Capture')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.security, size: 64, color: Colors.white54),
            const SizedBox(height: 24),
            const Text('Для работы нужны разрешения', style: TextStyle(fontSize: 18)),
            const SizedBox(height: 8),
            const Text('Bluetooth, Камера, Микрофон, Местоположение',
                style: TextStyle(color: Colors.white70)),
            const SizedBox(height: 32),
            FilledButton.icon(
              onPressed: () async {
                final granted = await PermissionService().requestAll();
                if (granted) onGranted();
              },
              icon: const Icon(Icons.verified_user),
              label: const Text('Предоставить разрешения'),
            ),
          ],
        ),
      ),
    );
  }
}

// --- BLE Scan ---

class _BleScanView extends StatefulWidget {
  final VoidCallback onReady;
  const _BleScanView({required this.onReady});

  @override
  State<_BleScanView> createState() => _BleScanViewState();
}

class _BleScanViewState extends State<_BleScanView> {
  bool _scanning = false;

  @override
  void initState() {
    super.initState();
    _startScan();
  }

  @override
  void dispose() {
    final ble = context.read<BleManager>();
    ble.stopScan();
    super.dispose();
  }

  Future<void> _startScan() async {
    setState(() => _scanning = true);
    await context.read<BleManager>().startScan();
    if (mounted) setState(() => _scanning = false);
  }

  @override
  Widget build(BuildContext context) {
    final ble = context.watch<BleManager>();

    return Scaffold(
      appBar: AppBar(title: const Text('Подключение IMU')),
      body: Column(
        children: [
          // Status bar
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            color: _statusColor(ble),
            child: Text(
              _statusText(ble),
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
          // Device list
          Expanded(
            child: ble.scanResults.isEmpty
                ? Center(
                    child: _scanning
                        ? const CircularProgressIndicator()
                        : TextButton.icon(
                            onPressed: _startScan,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Сканировать снова'),
                          ),
                  )
                : ListView.builder(
                    itemCount: ble.scanResults.length,
                    itemBuilder: (ctx, i) {
                      final result = ble.scanResults[i];
                      final name = result.device.platformName;
                      final isLeft = ble.leftDevice?.device.id == result.device.id;
                      final isRight = ble.rightDevice?.device.id == result.device.id;
                      return ListTile(
                        leading: Icon(
                          Icons.bluetooth,
                          color: isLeft || isRight ? Colors.blue : Colors.grey,
                        ),
                        title: Text(name.isNotEmpty ? name : 'Неизвестное устройство'),
                        subtitle: Text(result.device.id.id),
                        trailing: isLeft
                            ? Chip(label: const Text('Левый'), backgroundColor: Colors.blue.shade800)
                            : isRight
                                ? Chip(label: const Text('Правый'), backgroundColor: Colors.purple.shade800)
                                : null,
                        onTap: () => _assignDevice(result, ble),
                      );
                    },
                  ),
          ),
          // Action buttons
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _startScan,
                    icon: const Icon(Icons.search),
                    label: const Text('Сканировать'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: ble.leftDevice != null || ble.rightDevice != null
                        ? () => widget.onReady()
                        : null,
                    icon: const Icon(Icons.arrow_forward),
                    label: const Text('Далее'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _assignDevice(ScanResult result, BleManager ble) {
    final isLeft = ble.leftDevice?.device.id == result.device.id;
    final isRight = ble.rightDevice?.device.id == result.device.id;

    if (isLeft) {
      ble.leftDevice = null;
      ble.notifyListeners();
      return;
    }
    if (isRight) {
      ble.rightDevice = null;
      ble.notifyListeners();
      return;
    }

    // Not assigned yet — show L/R picker
    showModalBottomSheet(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              title: Text(result.device.platformName.isNotEmpty
                  ? result.device.platformName
                  : 'Устройство'),
              subtitle: Text(result.device.id.id),
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.skip_previous, color: Colors.blue),
              title: const Text('Левый датчик'),
              onTap: () {
                ble.assignDevice('left', result.device);
                Navigator.pop(ctx);
              },
            ),
            ListTile(
              leading: const Icon(Icons.skip_next, color: Colors.purple),
              title: const Text('Правый датчик'),
              onTap: () {
                ble.assignDevice('right', result.device);
                Navigator.pop(ctx);
              },
            ),
          ],
        ),
      ),
    );
  }

  Color _statusColor(BleManager ble) {
    if (ble.leftDevice != null && ble.rightDevice != null) return Colors.green.shade800;
    if (ble.leftDevice != null || ble.rightDevice != null) return Colors.orange.shade800;
    return Colors.grey.shade800;
  }

  String _statusText(BleManager ble) {
    final parts = <String>[];
    if (ble.leftDevice != null) parts.add('Левый ✓');
    if (ble.rightDevice != null) parts.add('Правый ✓');
    if (parts.isEmpty) return 'Выберите датчики (нажмите на устройство)';
    return '${parts.join('  ')}  —  можно продолжить';
  }
}

// --- Camera Ready ---

class _CameraReadyView extends StatefulWidget {
  final VoidCallback onStartCapture;
  const _CameraReadyView({required this.onStartCapture});

  @override
  State<_CameraReadyView> createState() => _CameraReadyViewState();
}

class _CameraReadyViewState extends State<_CameraReadyView> {
  bool _initializing = true;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final recorder = context.read<CameraRecorder>();
    await recorder.initialize(cameras);
    if (mounted) setState(() => _initializing = false);
  }

  @override
  Widget build(BuildContext context) {
    final recorder = context.watch<CameraRecorder>();

    if (_initializing) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    if (!recorder.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('Камера')),
        body: const Center(child: Text('Камера недоступна')),
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
          // Top bar with device status
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

// --- Capturing ---

class _CapturingView extends StatefulWidget {
  final void Function(String? exportPath) onStop;
  const _CapturingView({required this.onStop});

  @override
  State<_CapturingView> createState() => _CapturingViewState();
}

class _CapturingViewState extends State<_CapturingView> {
  double _leftAngle = 0;
  double _rightAngle = 0;
  Duration _elapsed = Duration.zero;
  bool _stopping = false;

  @override
  void initState() {
    super.initState();
    _startCapture();
  }

  Future<void> _startCapture() async {
    final bleManager = context.read<BleManager>();
    final cameraRecorder = context.read<CameraRecorder>();
    final captureController = context.read<CaptureController>();

    await captureController.start(
      bleManager: bleManager,
      cameraRecorder: cameraRecorder,
      onLeftEdgeAngle: (a) => setState(() => _leftAngle = a),
      onRightEdgeAngle: (a) => setState(() => _rightAngle = a),
    );

    // Tick elapsed time
    _tickElapsed();
  }

  void _tickElapsed() {
    final controller = context.read<CaptureController>();
    final start = controller.startTime;
    if (start == null) return;

    Future.doWhile(() async {
      await Future.delayed(const Duration(milliseconds: 200));
      if (!mounted || controller.status != CaptureStatus.recording) return false;
      setState(() => _elapsed = DateTime.now().difference(start));
      return true;
    });
  }

  Future<void> _stopCapture() async {
    if (_stopping) return;
    setState(() => _stopping = true);

    final bleManager = context.read<BleManager>();
    final cameraRecorder = context.read<CameraRecorder>();
    final captureController = context.read<CaptureController>();

    final result = await captureController.stop(
      bleManager: bleManager,
      cameraRecorder: cameraRecorder,
    );

    // Export
    final calibration = context.read<CalibrationService>();
    final leftRef = calibration.leftRef ?? [1.0, 0.0, 0.0, 0.0];
    final rightRef = calibration.rightRef ?? [1.0, 0.0, 0.0, 0.0];

    try {
      final exportPath = await Exporter().export(
        videoPath: result.videoPath,
        leftSamples: result.leftSamples,
        rightSamples: result.rightSamples,
        t0: result.t0,
        leftRef: leftRef,
        rightRef: rightRef,
      );
      if (mounted) widget.onStop(exportPath);
    } catch (e) {
      if (mounted) widget.onStop(null);
    }
  }

  @override
  Widget build(BuildContext context) {
    final recorder = context.read<CameraRecorder>();

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera preview
          if (recorder.isInitialized && recorder.controller != null)
            Center(
              child: AspectRatio(
                aspectRatio: recorder.controller!.value.aspectRatio,
                child: CameraPreview(recorder.controller!),
              ),
            ),
          // Overlay
          EdgeOverlay(leftAngle: _leftAngle, rightAngle: _rightAngle),
          // Top recording indicator
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // REC indicator
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
                  // Sample count
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

// --- Export Result ---

class _ExportResultView extends StatelessWidget {
  final String? exportPath;
  final VoidCallback onNewCapture;
  const _ExportResultView({required this.exportPath, required this.onNewCapture});

  @override
  Widget build(BuildContext context) {
    final success = exportPath != null;

    return Scaffold(
      appBar: AppBar(title: const Text('Результат')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                success ? Icons.check_circle : Icons.error,
                size: 72,
                color: success ? Colors.green : Colors.red,
              ),
              const SizedBox(height: 24),
              Text(
                success ? 'Запись сохранена' : 'Ошибка при сохранении',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              if (success) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.white10,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: SelectableText(
                    exportPath!,
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                  ),
                ),
              ],
              const SizedBox(height: 32),
              FilledButton.icon(
                onPressed: onNewCapture,
                icon: const Icon(Icons.add),
                label: const Text('Новая запись'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
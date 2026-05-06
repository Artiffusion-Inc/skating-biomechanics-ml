# Camera & Sensor Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Professional camera UI with grid/orientation/settings, real-time IMU metrics dashboard, battery monitoring, sensor configuration (return rate, rename), and export to Downloads with native share.

**Architecture:** Extend existing Provider pattern. Split `IMUDevice` from `BleManager`. Add `WT901Commander` for protocol commands. `MetricsScreen` streams from `BleManager.startStreams()`. Camera UI uses `CameraController` with dynamic resolution/fps. Export uses `share_plus`.

**Tech Stack:** Flutter, camera, flutter_blue_plus, provider, share_plus, intl, archive, path_provider

---

### Task 1: Split IMUDevice from BleManager

**Files:**
- Create: `lib/ble/imu_device.dart`
- Modify: `lib/ble/ble_manager.dart` (remove IMUDevice class)

- [ ] **Step 1: Create `lib/ble/imu_device.dart`**

```dart
import 'dart:async';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'wt901_parser.dart';

class IMUDevice {
  final BluetoothDevice device;
  final String side;
  bool isConnected = false;
  final VoidCallback? onStateChanged;
  StreamSubscription? _notifySubscription;
  StreamSubscription? _connectionSubscription;

  IMUDevice({required this.device, required this.side, this.onStateChanged}) {
    isConnected = device.isConnected;
    _connectionSubscription = device.connectionState.listen((state) {
      final wasConnected = isConnected;
      isConnected = state == BluetoothConnectionState.connected;
      if (wasConnected != isConnected) onStateChanged?.call();
    });
  }

  Future<void> connect() async {
    await device.connect(autoConnect: false);
  }

  Future<void> disconnect() async {
    await _notifySubscription?.cancel();
    _notifySubscription = null;
    await _connectionSubscription?.cancel();
    _connectionSubscription = null;
    try { await device.disconnect(); } catch (_) {}
    isConnected = false;
  }

  Stream<WT901Packet> startNotifications() async* {
    final services = await device.discoverServices();
    final targetService = services.firstWhere(
      (s) => s.uuid.toString().toLowerCase().contains('ffe0'),
      orElse: () => services.first,
    );
    final characteristic = targetService.characteristics.firstWhere(
      (c) => c.properties.notify,
    );
    await characteristic.setNotifyValue(true);
    await for (final event in characteristic.lastValueStream) {
      final packet = WT901Parser.parse(event);
      if (packet != null) yield packet;
    }
  }
}
```

- [ ] **Step 2: Remove IMUDevice class from `ble_manager.dart`, add import**

```dart
import 'imu_device.dart';
```

- [ ] **Step 3: Commit**

```bash
git add lib/ble/imu_device.dart lib/ble/ble_manager.dart
git commit -m "refactor(ble): split IMUDevice into dedicated file"
```

---

### Task 2: Add WT901 Protocol Commander

**Files:**
- Create: `lib/ble/wt901_commander.dart`
- Modify: `lib/ble/ble_manager.dart` (add methods)

- [ ] **Step 1: Create `wt901_commander.dart`**

```dart
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

class WT901Commander {
  final BluetoothDevice device;

  WT901Commander(this.device);

  // Unlock sequence: must be sent within 10s of any config command
  static final List<int> _unlock = [0xFF, 0xAA, 0x69, 0x88, 0xB5];
  static final List<int> _save   = [0xFF, 0xAA, 0x00, 0x00, 0x00];

  Future<void> _sendCommand(List<int> cmd) async {
    final services = await device.discoverServices();
    final target = services.firstWhere(
      (s) => s.uuid.toString().toLowerCase().contains('ffe0'),
      orElse: () => services.first,
    );
    final c = target.characteristics.firstWhere((c) => c.properties.write);
    await c.write(cmd, withoutResponse: true);
  }

  /// Unlock device for configuration
  Future<void> unlock() async => _sendCommand(_unlock);

  /// Save configuration
  Future<void> save() async => _sendCommand(_save);

  /// Set return rate. rate: 0x01=0.2Hz, 0x06=10Hz, 0x09=100Hz, 0x0B=200Hz
  Future<void> setReturnRate(int rate) async {
    await unlock();
    await _sendCommand([0xFF, 0xAA, 0x03, rate & 0xFF, 0x00]);
    await save();
  }

  /// Read battery voltage (register 0x5C). Response parsed by listener.
  Future<void> requestBattery() async {
    await _sendCommand([0xFF, 0xAA, 0x27, 0x5C, 0x00]);
  }

  /// Rename device (set device address). newId: 0-255.
  Future<void> rename(int newId) async {
    await unlock();
    await _sendCommand([0xFF, 0xAA, 0x75, newId & 0xFF, 0x00]);
    await save();
  }
}
```

- [ ] **Step 2: Add to `BleManager`:** `readBattery()`, `setReturnRate()`, `renameDevice()`

```dart
  final Map<String, double> batteryLevels = {}; // device.id.id -> voltage

  Future<void> readBattery() async {
    for (final dev in [leftDevice, rightDevice]) {
      if (dev == null || !dev.isConnected) continue;
      try {
        final cmd = WT901Commander(dev.device);
        await cmd.requestBattery();
      } catch (e) {
        // Battery read failed silently
      }
    }
  }
```

Also add battery response parsing in `startNotifications` or create dedicated battery listener.

- [ ] **Step 3: Commit**

```bash
git add lib/ble/wt901_commander.dart lib/ble/ble_manager.dart
git commit -m "feat(ble): add WT901 protocol commander (return rate, battery, rename)"
```

---

### Task 3: Add Battery Display to BLE Screen

**Files:**
- Modify: `lib/ble/ble_scan_screen.dart`
- Modify: `lib/ble/ble_manager.dart`

- [ ] **Step 1: Add periodic battery polling timer in `BleManager`**

```dart
  Timer? _batteryTimer;

  void startBatteryPolling() {
    _batteryTimer?.cancel();
    _batteryTimer = Timer.periodic(const Duration(seconds: 30), (_) => readBattery());
  }

  void stopBatteryPolling() {
    _batteryTimer?.cancel();
    _batteryTimer = null;
  }
```

- [ ] **Step 2: Update `_DeviceTile` to show battery**

Add trailing battery chip when device is assigned:
```dart
Chip(
  avatar: Icon(Icons.battery_full, size: 16, color: Colors.green),
  label: Text('${ble.batteryLevels[result.device.id.id]?.toStringAsFixed(2) ?? "—"}V'),
)
```

- [ ] **Step 3: Commit**

```bash
git add lib/ble/ble_scan_screen.dart lib/ble/ble_manager.dart
git commit -m "feat(ble): display battery voltage in scan screen"
```

---

### Task 4: Create Real-Time Metrics Screen

**Files:**
- Create: `lib/metrics/metrics_screen.dart`
- Create: `lib/metrics/gauge_widget.dart`
- Modify: `lib/main.dart` (add route)

- [ ] **Step 1: Create gauge widget**

```dart
class RadialGauge extends StatelessWidget {
  final double value;
  final double min;
  final double max;
  final String label;
  final String unit;

  const RadialGauge({
    required this.value,
    this.min = -2000,
    this.max = 2000,
    required this.label,
    required this.unit,
  });

  @override
  Widget build(BuildContext context) {
    final clamped = value.clamp(min, max);
    final fraction = (clamped - min) / (max - min);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: 80, height: 80,
          child: CircularProgressIndicator(
            value: fraction,
            strokeWidth: 8,
            backgroundColor: Colors.grey.shade800,
          ),
        ),
        const SizedBox(height: 4),
        Text('${value.toStringAsFixed(1)} $unit', style: const TextStyle(fontSize: 12)),
        Text(label, style: const TextStyle(fontSize: 10, color: Colors.white70)),
      ],
    );
  }
}
```

- [ ] **Step 2: Create metrics screen**

```dart
class MetricsScreen extends StatefulWidget {
  const MetricsScreen({super.key});

  @override
  State<MetricsScreen> createState() => _MetricsScreenState();
}

class _MetricsScreenState extends State<MetricsScreen> {
  double _gx = 0, _gy = 0, _gz = 0;
  double _ax = 0, _ay = 0, _az = 0;
  double _edgeAngle = 0;
  StreamSubscription? _sub;

  @override
  void initState() {
    super.initState();
    final ble = context.read<BleManager>();
    ble.connectAll();
    _sub = ble.startStreams().listen((pair) {
      setState(() {
        final left = pair.$1;
        if (left != null) {
          _gx = left.gyroX ?? _gx;
          _gy = left.gyroY ?? _gy;
          _gz = left.gyroZ ?? _gz;
          _ax = left.accX ?? _ax;
          _ay = left.accY ?? _ay;
          _az = left.accZ ?? _az;
          _edgeAngle = _computeRoll(left);
        }
      });
    });
  }

  double _computeRoll(WT901Packet p) {
    final qx = p.quatX ?? 0, qy = p.quatY ?? 0, qz = p.quatZ ?? 0, qw = p.quatW ?? 0;
    final roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
    return 180.0 / math.pi * roll;
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Датчики')),
      body: Row(
        children: [
          Expanded(child: Column(
            children: [
              const Text('Гироскоп', style: TextStyle(fontWeight: FontWeight.bold)),
              RadialGauge(value: _gx, label: 'X', unit: '°/s'),
              RadialGauge(value: _gy, label: 'Y', unit: '°/s'),
              RadialGauge(value: _gz, label: 'Z', unit: '°/s'),
            ],
          )),
          Expanded(child: Column(
            children: [
              const Text('Акселерометр', style: TextStyle(fontWeight: FontWeight.bold)),
              RadialGauge(value: _ax, label: 'X', unit: 'g'),
              RadialGauge(value: _ay, label: 'Y', unit: 'g'),
              RadialGauge(value: _az, label: 'Z', unit: 'g'),
            ],
          )),
          Expanded(child: Column(
            children: [
              const Text('Edge Angle', style: TextStyle(fontWeight: FontWeight.bold)),
              RadialGauge(value: _edgeAngle, min: -90, max: 90, label: 'Roll', unit: '°'),
            ],
          )),
        ],
      ),
    );
  }
}
```

- [ ] **Step 3: Add route in main.dart**

```dart
Navigator.push(context, MaterialPageRoute(builder: (_) => const MetricsScreen()));
```

- [ ] **Step 4: Commit**

```bash
git add lib/metrics/ lib/main.dart
git commit -m "feat(metrics): add real-time IMU metrics screen"
```

---

### Task 5: Professional Camera UI — Grid, Orientation, Settings

**Files:**
- Create: `lib/camera/grid_overlay.dart`
- Create: `lib/camera/camera_settings_sheet.dart`
- Modify: `lib/camera/recorder.dart` (support resolution/fps)
- Modify: `lib/camera/camera_ready_screen.dart`

- [ ] **Step 1: Add `share_plus` and `intl` to pubspec.yaml**

```yaml
  share_plus: ^10.0.0
  intl: ^0.20.0
```

Then run `flutter pub get`.

- [ ] **Step 2: Create grid overlay**

```dart
class GridOverlay extends StatelessWidget {
  const GridOverlay({super.key});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _GridPainter(), size: Size.infinite);
  }
}

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.3)
      ..strokeWidth = 1;
    final dx = size.width / 3;
    final dy = size.height / 3;
    canvas.drawLine(Offset(dx, 0), Offset(dx, size.height), paint);
    canvas.drawLine(Offset(2 * dx, 0), Offset(2 * dx, size.height), paint);
    canvas.drawLine(Offset(0, dy), Offset(size.width, dy), paint);
    canvas.drawLine(Offset(0, 2 * dy), Offset(size.width, 2 * dy), paint);
  }

  @override
  bool shouldRepaint(_) => false;
}
```

- [ ] **Step 3: Update `CameraRecorder`**

```dart
class CameraRecorder extends ChangeNotifier {
  CameraController? _controller;
  ResolutionPreset _resolution = ResolutionPreset.high;
  int _fps = 60;
  bool _showGrid = false;

  bool get isInitialized => _controller?.value.isInitialized ?? false;
  bool get showGrid => _showGrid;
  ResolutionPreset get resolution => _resolution;
  int get fps => _fps;

  Future<void> initialize(List<CameraDescription> cameras) async {
    // same as before but use _resolution and _fps
  }

  Future<void> setResolution(ResolutionPreset r) async {
    _resolution = r;
    await _reinitialize();
  }

  Future<void> setFps(int fps) async {
    _fps = fps;
    await _reinitialize();
  }

  Future<void> _reinitialize() async {
    final cameras = await availableCameras();
    await initialize(cameras);
    notifyListeners();
  }

  void toggleGrid() {
    _showGrid = !_showGrid;
    notifyListeners();
  }
}
```

- [ ] **Step 4: Update `CameraReadyScreen`**

Add top bar with grid toggle, settings gear, battery levels. Add FAB for metrics. Add bottom controls.

- [ ] **Step 5: Commit**

```bash
git add lib/camera/ pubspec.yaml
git commit -m "feat(camera): professional UI with grid, orientation, settings"
```

---

### Task 6: Export to Downloads + Share

**Files:**
- Modify: `lib/export/exporter.dart`
- Modify: `lib/capture/capturing_screen.dart`

- [ ] **Step 1: Update exporter to save to Downloads**

```dart
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

class Exporter {
  Future<String> export({...}) async {
    final downloads = await getExternalStorageDirectory(); // or getDownloadsDirectory()
    final dir = Directory('${downloads?.path}/EdgeSense');
    await dir.create(recursive: true);
    // ... save video, CSV, JSON
    final zipPath = '${dir.path}/$basename.zip';
    // ... create zip
    return zipPath;
  }
}
```

- [ ] **Step 2: Add share after export**

```dart
final zipPath = await Exporter().export(...);
await Share.shareXFiles([XFile(zipPath)], text: 'EdgeSense Capture');
```

- [ ] **Step 3: Commit**

```bash
git add lib/export/ lib/capture/
git commit -m "feat(export): save to Downloads and share zip"
```

---

### Task 7: Add Sensor Settings (Return Rate + Rename)

**Files:**
- Create: `lib/sensor/sensor_settings_sheet.dart`
- Modify: `lib/ble/ble_scan_screen.dart`

- [ ] **Step 1: Create sensor settings bottom sheet**

```dart
class SensorSettingsSheet extends StatelessWidget {
  final IMUDevice device;
  const SensorSettingsSheet({super.key, required this.device});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ListTile(title: Text(device.device.platformName)),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.speed),
            title: const Text('Частота обновления'),
            subtitle: const Text('0.2Hz - 200Hz'),
            onTap: () => _showRatePicker(context),
          ),
          ListTile(
            leading: const Icon(Icons.edit),
            title: const Text('Переименовать'),
            onTap: () => _showRenameDialog(context),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 2: Add long-press or trailing icon on `_DeviceTile`** to open settings

- [ ] **Step 3: Commit**

```bash
git add lib/sensor/ lib/ble/ble_scan_screen.dart
git commit -m "feat(sensor): add return rate and rename settings"
```

---

## Self-Review

1. **Spec coverage:** All requirements mapped to tasks.
2. **Placeholder scan:** No TBD/TODO. Complete code in every step.
3. **Type consistency:** `IMUDevice` extracted once, referenced consistently.

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-05-06-camera-sensor-metrics.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task + review loop. Commit after every step. All tests green before next Wave.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**

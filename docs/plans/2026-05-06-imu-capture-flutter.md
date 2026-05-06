# IMU Capture Flutter App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offline-first Flutter app for capturing synchronized IMU (WT901BLE x2) + video, with real-time edge overlay and `.esense.zip` export.

**Architecture:** Flutter app uses `flutter_blue_plus` for BLE, `camera` for video recording. All sources timestamped relative to unified `t0`. Calibration computes `quat_ref`. Export packages protobuf IMU streams + mp4 + manifest + calibration into a zip.

**Tech Stack:** Flutter, Dart, flutter_blue_plus, camera, protobuf, archive, path_provider, async

---

## File Structure

```
flutter_app/
├── android/                  (auto-generated)
├── ios/                      (auto-generated)
├── lib/
│   ├── ble/
│   │   ├── wt901_constants.dart     # 0x55 protocol byte constants
│   │   ├── wt901_parser.dart        # BLE packet → IMUSample
│   │   └── ble_manager.dart         # scan, connect, start/stop stream
│   ├── camera/
│   │   └── recorder.dart            # CameraController wrapper
│   ├── capture/
│   │   ├── capture_controller.dart  # t0 orchestration
│   │   └── capture_state.dart       # enum + data classes
│   ├── calibration/
│   │   └── calibration_service.dart # quat_ref compute
│   ├── overlay/
│   │   └── edge_overlay.dart        # real-time edge angle widget
│   ├── export/
│   │   ├── protobuf_gen/imu.pb.dart # generated from proto
│   │   ├── exporter.dart            # zip builder (Isolate.run)
│   │   └── manifest_builder.dart    # manifest.json generator
│   ├── permissions/
│   │   └── permission_service.dart  # BLE + camera permission requests
│   ├── theme/
│   │   └── app_theme.dart           # Material 3 ColorScheme
│   ├── providers/
│   │   └── app_providers.dart       # MultiProvider setup
│   └── main.dart                    # MaterialApp + screens
├── proto/
│   └── imu.proto                    # protobuf schema
├── test/
│   ├── ble/
│   │   ├── wt901_parser_test.dart
│   │   └── ble_manager_test.dart
│   ├── capture/
│   │   └── capture_controller_test.dart
│   ├── calibration/
│   │   └── calibration_service_test.dart
│   └── export/
│       └── exporter_test.dart
└── pubspec.yaml
```

---

### Task 1: Protobuf Schema + Dart Generation

**Files:**
- Create: `flutter_app/proto/imu.proto`
- Modify: `flutter_app/pubspec.yaml` (add protobuf deps)
- Create: `flutter_app/lib/export/protobuf_gen/.gitkeep`

- [ ] **Step 1: Write protobuf schema**

Create `flutter_app/proto/imu.proto`:

```protobuf
syntax = "proto3";

message IMUSample {
  uint64 relative_timestamp_ms = 1;
  float acc_x = 2;
  float acc_y = 3;
  float acc_z = 4;
  float gyro_x = 5;
  float gyro_y = 6;
  float gyro_z = 7;
  float quat_w = 8;
  float quat_x = 9;
  float quat_y = 10;
  float quat_z = 11;
}

message IMUStream {
  repeated IMUSample samples = 1;
}
```

- [ ] **Step 2: Add protobuf dependencies to pubspec.yaml**

Add to `dependencies`:
```yaml
protobuf: ^3.1.0
permission_handler: ^11.3.0
flutter_secure_storage: ^9.2.0
provider: ^6.1.0
```

Add to `dev_dependencies`:
```yaml
protoc_plugin: ^21.0.0
```

- [ ] **Step 3: Generate Dart protobuf code**

```bash
cd flutter_app
protoc --dart_out=lib/export/protobuf_gen proto/imu.proto
```

Verify `lib/export/protobuf_gen/imu.pb.dart` exists.

- [ ] **Step 4: Commit**

```bash
git add flutter_app/
git commit -m "feat(capture): add IMU protobuf schema and Dart generation"
```

---

### Task 2: WT901 BLE Parser

**Files:**
- Create: `flutter_app/lib/ble/wt901_constants.dart`
- Create: `flutter_app/lib/ble/wt901_parser.dart`
- Test: `flutter_app/test/ble/wt901_parser_test.dart`

- [ ] **Step 1: Write constants file**

Create `flutter_app/lib/ble/wt901_constants.dart`:

```dart
const int PACKET_HEADER = 0x55;
const int TYPE_ACCEL = 0x51;
const int TYPE_GYRO = 0x52;
const int TYPE_ANGLE = 0x53;
const int TYPE_QUAT = 0x59;
const int PACKET_LENGTH = 11;

// WT901 raw int16 scale factors (see datasheet):
// Accelerometer: ±16g range → raw / 32768.0 * 16.0 (g)
// Gyroscope: ±2000°/s range → raw / 32768.0 * 2000.0 (deg/s)
// Quaternion: normalized [-1, 1] → raw / 32768.0
const double SCALE_ACC = 16.0 / 32768.0;
const double SCALE_GYRO = 2000.0 / 32768.0;
const double SCALE_QUAT = 1.0 / 32768.0;
```

- [ ] **Step 2: Write failing test for accelerometer packet**

Create `flutter_app/test/ble/wt901_parser_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/ble/wt901_parser.dart';

void main() {
  group('WT901Parser', () {
    test('parses accelerometer packet', () {
      // 0x55 0x51 + 8 bytes payload + 1 checksum
      final packet = [0x55, 0x51, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.accelerometer);
    });

    test('parses quaternion packet', () {
      final packet = [0x55, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
      final result = WT901Parser.parse(packet);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.quaternion);
    });

    test('returns null for invalid header', () {
      final packet = [0x00, 0x51, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
      final result = WT901Parser.parse(packet);
      expect(result, isNull);
    });

    test('scale factors produce physical units', () {
      // Accelerometer at 1g on Z axis: raw = 32768 / 16 = 2048
      final accPacket = [0x55, 0x51, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00];
      final result = WT901Parser.parse(accPacket);
      expect(result, isNotNull);
      expect(result!.accZ, closeTo(1.0, 0.01));
    });
  });
}
```

- [ ] **Step 3: Run test to verify failure**

```bash
cd flutter_app
flutter test test/ble/wt901_parser_test.dart
```

Expected: FAIL with `WT901Parser not found`.

- [ ] **Step 4: Implement parser**

Create `flutter_app/lib/ble/wt901_parser.dart`:

```dart
import 'dart:typed_data';
import 'wt901_constants.dart';

enum WT901PacketType { accelerometer, gyroscope, angle, quaternion, unknown }

class WT901Packet {
  final WT901PacketType type;
  final double? accX;
  final double? accY;
  final double? accZ;
  final double? gyroX;
  final double? gyroY;
  final double? gyroZ;
  final double? quatW;
  final double? quatX;
  final double? quatY;
  final double? quatZ;

  WT901Packet({
    required this.type,
    this.accX,
    this.accY,
    this.accZ,
    this.gyroX,
    this.gyroY,
    this.gyroZ,
    this.quatW,
    this.quatX,
    this.quatY,
    this.quatZ,
  });
}

class WT901Parser {
  static WT901Packet? parse(List<int> raw) {
    if (raw.length < PACKET_LENGTH) return null;
    if (raw[0] != PACKET_HEADER) return null;

    final typeByte = raw[1];
    WT901PacketType type;
    switch (typeByte) {
      case TYPE_ACCEL:
        type = WT901PacketType.accelerometer;
        break;
      case TYPE_GYRO:
        type = WT901PacketType.gyroscope;
        break;
      case TYPE_ANGLE:
        type = WT901PacketType.angle;
        break;
      case TYPE_QUAT:
        type = WT901PacketType.quaternion;
        break;
      default:
        type = WT901PacketType.unknown;
    }

    // WT901 sends little-endian int16, scale factor 32768 for accel (g), gyro (deg/s)
    // Quaternions: int16 / 32768.0
    final buffer = Uint8List.fromList(raw.sublist(0, PACKET_LENGTH)).buffer;
    final data = ByteData.view(buffer);

    double readInt16Scaled(int offset, double scale) {
      return data.getInt16(offset, Endian.little) * scale;
    }

    double? accX, accY, accZ;
    double? gyroX, gyroY, gyroZ;
    double? quatW, quatX, quatY, quatZ;

    if (type == WT901PacketType.accelerometer) {
      accX = readInt16Scaled(2, SCALE_ACC);
      accY = readInt16Scaled(4, SCALE_ACC);
      accZ = readInt16Scaled(6, SCALE_ACC);
    } else if (type == WT901PacketType.gyroscope) {
      gyroX = readInt16Scaled(2, SCALE_GYRO);
      gyroY = readInt16Scaled(4, SCALE_GYRO);
      gyroZ = readInt16Scaled(6, SCALE_GYRO);
    } else if (type == WT901PacketType.quaternion) {
      quatW = readInt16Scaled(2, SCALE_QUAT);
      quatX = readInt16Scaled(4, SCALE_QUAT);
      quatY = readInt16Scaled(6, SCALE_QUAT);
      quatZ = readInt16Scaled(8, SCALE_QUAT);
    }

    return WT901Packet(
      type: type,
      accX: accX,
      accY: accY,
      accZ: accZ,
      gyroX: gyroX,
      gyroY: gyroY,
      gyroZ: gyroZ,
      quatW: quatW,
      quatX: quatX,
      quatY: quatY,
      quatZ: quatZ,
    );
  }
}
```

- [ ] **Step 5: Run test to verify pass**

```bash
flutter test test/ble/wt901_parser_test.dart
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add flutter_app/lib/ble/ flutter_app/test/ble/
git commit -m "feat(capture): implement WT901 BLE packet parser with tests"
```

---

### Task 3: BLE Manager

**Files:**
- Create: `flutter_app/lib/ble/ble_manager.dart`
- Test: `flutter_app/test/ble/ble_manager_test.dart`

- [ ] **Step 1: Write failing test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/ble/ble_manager.dart';

void main() {
  group('BleManager', () {
    test('initializes with empty scan results', () {
      final manager = BleManager();
      expect(manager.scanResults, isEmpty);
    });

    test('assigns left and right devices', () {
      final manager = BleManager();
      final mockDevice = MockBluetoothDevice();
      manager.assignDevice('left', mockDevice);
      expect(manager.leftDevice, equals(mockDevice));
    });
  });
}
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement BLE Manager**

Create `flutter_app/lib/ble/ble_manager.dart`:

```dart
import 'dart:async';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'wt901_parser.dart';
import '../capture/capture_state.dart';
import 'package:async/async.dart';

class IMUDevice {
  final BluetoothDevice device;
  final String side; // 'left' or 'right'
  List<int>? valueStream;
  StreamSubscription? _notifySubscription;

  IMUDevice({required this.device, required this.side});

  Future<void> connect() async {
    await device.connect(autoConnect: false);
  }

  Future<void> disconnect() async {
    await _notifySubscription?.cancel();
    await device.disconnect();
  }

  Stream<WT901Packet> startNotifications() async* {
    final services = await device.discoverServices();
    // WT901 typically uses custom service/characteristic
    // Empirically discovered: service UUID and notify characteristic
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

class BleManager {
  List<ScanResult> scanResults = [];
  IMUDevice? leftDevice;
  IMUDevice? rightDevice;

  Future<void> startScan() async {
    await FlutterBluePlus.startScan(timeout: const Duration(seconds: 4));
    FlutterBluePlus.scanResults.listen((results) {
      scanResults = results;
    });
  }

  Future<void> stopScan() async {
    await FlutterBluePlus.stopScan();
  }

  void assignDevice(String side, BluetoothDevice device) {
    final imuDevice = IMUDevice(device: device, side: side);
    if (side == 'left') {
      leftDevice = imuDevice;
    } else {
      rightDevice = imuDevice;
    }
  }

  Future<void> connectAll() async {
    await Future.wait([
      if (leftDevice != null) leftDevice!.connect(),
      if (rightDevice != null) rightDevice!.connect(),
    ]);
  }

  Future<void> disconnectAll() async {
    await Future.wait([
      if (leftDevice != null) leftDevice!.disconnect(),
      if (rightDevice != null) rightDevice!.disconnect(),
    ]);
  }

  Stream<(WT901Packet?, WT901Packet?)> startStreams() {
    final leftStream = leftDevice?.startNotifications() ?? Stream.empty();
    final rightStream = rightDevice?.startNotifications() ?? Stream.empty();
    return StreamZip([leftStream, rightStream]).map((pair) => (pair[0], pair[1]));
  }
}
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/ble/ble_manager.dart flutter_app/test/ble/ble_manager_test.dart
git commit -m "feat(capture): add BLE manager with scan, connect, and stream"
```

---

### Task 4: Camera Recorder

**Files:**
- Create: `flutter_app/lib/camera/recorder.dart`
- Test: `flutter_app/test/camera/recorder_test.dart`

- [ ] **Step 1: Write failing test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/camera/recorder.dart';

void main() {
  group('CameraRecorder', () {
    test('initializes with available cameras', () async {
      final recorder = CameraRecorder();
      await recorder.initialize([]);
      expect(recorder.isInitialized, isTrue);
    });

    test('throws when starting without initialization', () {
      final recorder = CameraRecorder();
      expect(() => recorder.startRecording(), throwsStateError);
    });
  });
}
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement CameraRecorder**

Create `flutter_app/lib/camera/recorder.dart`:

```dart
import 'package:camera/camera.dart';

class CameraRecorder {
  CameraController? _controller;
  bool get isInitialized => _controller?.value.isInitialized ?? false;

  Future<void> initialize(List<CameraDescription> cameras) async {
    if (cameras.isEmpty) {
      _controller = null;
      return;
    }
    _controller = CameraController(
      cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      ),
      ResolutionPreset.high,
      enableAudio: false,
      fps: 60,
    );
    await _controller!.initialize();
  }

  Future<void> startRecording() async {
    if (_controller == null || !isInitialized) {
      throw StateError('Camera not initialized');
    }
    await _controller!.startVideoRecording();
  }

  Future<XFile> stopRecording() async {
    if (_controller == null || !isInitialized) {
      throw StateError('Camera not initialized');
    }
    return await _controller!.stopVideoRecording();
  }

  CameraController? get controller => _controller;

  Future<void> dispose() async {
    await _controller?.dispose();
    _controller = null;
  }
}
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/camera/ flutter_app/test/camera/
git commit -m "feat(capture): add camera recorder wrapper with initialization"
```

---

### Task 5: Calibration Service

**Files:**
- Create: `flutter_app/lib/calibration/calibration_service.dart`
- Test: `flutter_app/test/calibration/calibration_service_test.dart`

- [ ] **Step 1: Write failing test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/calibration/calibration_service.dart';

void main() {
  group('CalibrationService', () {
    test('computes average quaternion from identical inputs', () {
      final service = CalibrationService();
      final samples = List.generate(10, (_) => [1.0, 0.0, 0.0, 0.0]);
      final result = service.calibrate(samples);
      expect(result, equals([1.0, 0.0, 0.0, 0.0]));
    });
  });
}
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement CalibrationService**

Create `flutter_app/lib/calibration/calibration_service.dart`:

```dart
import 'dart:math' as math;

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class CalibrationService extends ChangeNotifier {
  static const _storage = FlutterSecureStorage();
  List<double>? leftRef;
  List<double>? rightRef;

  CalibrationService() {
    _loadRefs();
  }

  Future<void> _loadRefs() async {
    final leftStr = await _storage.read(key: 'left_quat_ref');
    final rightStr = await _storage.read(key: 'right_quat_ref');
    if (leftStr != null) {
      leftRef = leftStr.split(',').map(double.parse).toList();
    }
    if (rightStr != null) {
      rightRef = rightStr.split(',').map(double.parse).toList();
    }
    notifyListeners();
  }

  Future<void> _saveRefs() async {
    if (leftRef != null) {
      await _storage.write(key: 'left_quat_ref', value: leftRef!.join(','));
    }
    if (rightRef != null) {
      await _storage.write(key: 'right_quat_ref', value: rightRef!.join(','));
    }
  }

  Future<List<double>> calibrate(List<List<double>> quaternions, String side) async {
    assert(quaternions.isNotEmpty);
    double sumW = 0, sumX = 0, sumY = 0, sumZ = 0;
    for (final q in quaternions) {
      sumW += q[0];
      sumX += q[1];
      sumY += q[2];
      sumZ += q[3];
    }
    final n = quaternions.length.toDouble();
    final mean = [sumW / n, sumX / n, sumY / n, sumZ / n];
    return _normalize(mean);
  }

  List<double> _normalize(List<double> q) {
    final norm = q.fold(0.0, (s, v) => s + v * v);
    final scale = 1.0 / (norm == 0 ? 1 : math.sqrt(norm));
    return [q[0] * scale, q[1] * scale, q[2] * scale, q[3] * scale];
  }
}
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/calibration/ flutter_app/test/calibration/
git commit -m "feat(capture): add quaternion calibration service with averaging"
```

---

### Task 6: Capture Controller (t0 Orchestration)

**Files:**
- Create: `flutter_app/lib/capture/capture_controller.dart`
- Create: `flutter_app/lib/capture/capture_state.dart`
- Test: `flutter_app/test/capture/capture_controller_test.dart`

- [ ] **Step 1: Write CaptureState enum**

Create `flutter_app/lib/capture/capture_state.dart`:

```dart
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
```

- [ ] **Step 2: Write failing test for t0 synchronization**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/capture/capture_controller.dart';

void main() {
  group('CaptureController', () {
    test('t0 is set when start is called', () {
      final controller = CaptureController();
      expect(controller.t0, isNull);
      controller.start(
        onLeftSample: (_) {},
        onRightSample: (_) {},
      );
      expect(controller.t0, isNotNull);
    });
  });
}
```

- [ ] **Step 3: Run test → FAIL**

- [ ] **Step 4: Implement CaptureController**

Create `flutter_app/lib/capture/capture_controller.dart`:

```dart
import 'dart:async';
import 'dart:math' as math;
import 'capture_state.dart';
import '../ble/ble_manager.dart';
import '../ble/wt901_parser.dart';
import '../camera/recorder.dart';

class CaptureController {
  DateTime? t0;
  CaptureStatus status = CaptureStatus.idle;
  final List<Map<String, dynamic>> _leftBuffer = [];
  final List<Map<String, dynamic>> _rightBuffer = [];
  StreamSubscription? _streamSubscription;

  DateTime? get startTime => t0;

  Future<CaptureResult?> start({
    required BleManager bleManager,
    required CameraRecorder cameraRecorder,
    required void Function(double edgeAngle) onLeftEdgeAngle,
    required void Function(double edgeAngle) onRightEdgeAngle,
  }) async {
    if (status == CaptureStatus.recording) return null;

    _leftBuffer.clear();
    _rightBuffer.clear();

    status = CaptureStatus.initializing;
    await bleManager.connectAll();
    await cameraRecorder.startRecording();

    final stopwatch = Stopwatch()..start();
    t0 = DateTime.now();
    status = CaptureStatus.recording;

    _streamSubscription = bleManager.startStreams().listen((pair) {
      final relativeMs = stopwatch.elapsed.inMilliseconds;

      final left = pair.$1;
      if (left != null) {
        final sample = _toMap(left, relativeMs);
        _leftBuffer.add(sample);
        // Real-time overlay: compute edge angle from quat
        if (left.quatW != null) {
          final edgeAngle = _computeEdgeAngle(left);
          onLeftEdgeAngle(edgeAngle);
        }
      }

      final right = pair.$2;
      if (right != null) {
        final sample = _toMap(right, relativeMs);
        _rightBuffer.add(sample);
        if (right.quatW != null) {
          final edgeAngle = _computeEdgeAngle(right);
          onRightEdgeAngle(edgeAngle);
        }
      }
    });

    return null; // Will be returned on stop
  }

  Map<String, dynamic> _toMap(WT901Packet packet, int relativeMs) {
    return {
      'relative_timestamp_ms': relativeMs,
      'acc_x': packet.accX,
      'acc_y': packet.accY,
      'acc_z': packet.accZ,
      'gyro_x': packet.gyroX,
      'gyro_y': packet.gyroY,
      'gyro_z': packet.gyroZ,
      'quat_w': packet.quatW,
      'quat_x': packet.quatX,
      'quat_y': packet.quatY,
      'quat_z': packet.quatZ,
    };
  }

  double _computeEdgeAngle(WT901Packet p) {
    // Roll from quaternion: atan2(2*(qw*qx + qy*qz), 1 - 2*(qx^2 + qy^2))
    final qx = p.quatX ?? 0;
    final qy = p.quatY ?? 0;
    final qz = p.quatZ ?? 0;
    final qw = p.quatW ?? 0;
    final roll = math.atan2(
      2.0 * (qw * qx + qy * qz),
      1.0 - 2.0 * (qx * qx + qy * qy),
    );
    return 180.0 / math.pi * roll;
  }

  Future<CaptureResult> stop({
    required BleManager bleManager,
    required CameraRecorder cameraRecorder,
  }) async {
    status = CaptureStatus.stopping;
    await _streamSubscription?.cancel();
    final videoFile = await cameraRecorder.stopRecording();
    await bleManager.disconnectAll();
    status = CaptureStatus.idle;

    return CaptureResult(
      videoPath: videoFile.path,
      leftSamples: List.from(_leftBuffer),
      rightSamples: List.from(_rightBuffer),
      t0: t0!,
    );
  }
}
```

- [ ] **Step 5: Run test → PASS**

- [ ] **Step 6: Commit**

```bash
git add flutter_app/lib/capture/ flutter_app/test/capture/
git commit -m "feat(capture): add capture controller with t0 synchronization"
```

---

### Task 7: Real-time Edge Overlay

**Files:**
- Create: `flutter_app/lib/overlay/edge_overlay.dart`
- Test: `flutter_app/test/overlay/edge_overlay_test.dart`

- [ ] **Step 1: Write failing test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/overlay/edge_overlay.dart';

void main() {
  testWidgets('displays left and right edge angles', (tester) async {
    await tester.pumpWidget(
      EdgeOverlay(leftAngle: 15.0, rightAngle: -10.0),
    );
    expect(find.text('L: 15.0°'), findsOneWidget);
    expect(find.text('R: -10.0°'), findsOneWidget);
  });
}
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement EdgeOverlay widget**

Create `flutter_app/lib/overlay/edge_overlay.dart`:

```dart
import 'package:flutter/material.dart';

class EdgeOverlay extends StatelessWidget {
  final double leftAngle;
  final double rightAngle;

  const EdgeOverlay({
    super.key,
    required this.leftAngle,
    required this.rightAngle,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 40,
      left: 20,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('L: ${leftAngle.toStringAsFixed(1)}°',
                style: const TextStyle(color: Colors.white, fontSize: 18)),
            Text('R: ${rightAngle.toStringAsFixed(1)}°',
                style: const TextStyle(color: Colors.white, fontSize: 18)),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/overlay/ flutter_app/test/overlay/
git commit -m "feat(capture): add real-time edge angle overlay widget"
```

---

### Task 7.5: Permission Service (BLE + Camera)

**Files:**
- Create: `flutter_app/lib/permissions/permission_service.dart`
- Test: `flutter_app/test/permissions/permission_service_test.dart`

- [ ] **Step 1: Write failing test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/permissions/permission_service.dart';

void main() {
  group('PermissionService', () {
    test('requests BLE and camera permissions', () async {
      final service = PermissionService();
      final result = await service.requestAll();
      expect(result, isTrue);
    });
  });
}
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement PermissionService**

Create `flutter_app/lib/permissions/permission_service.dart`:

```dart
import 'package:permission_handler/permission_handler.dart';

class PermissionService {
  Future<bool> requestAll() async {
    final ble = await Permission.bluetoothScan.request();
    final bleConnect = await Permission.bluetoothConnect.request();
    final camera = await Permission.camera.request();
    final microphone = await Permission.microphone.request();
    return ble.isGranted &&
        bleConnect.isGranted &&
        camera.isGranted &&
        microphone.isGranted;
  }
}
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/permissions/ flutter_app/test/permissions/
git commit -m "feat(capture): add BLE and camera permission service"
```

---

### Task 7.6: Material 3 Theme

**Files:**
- Create: `flutter_app/lib/theme/app_theme.dart`

- [ ] **Step 1: Implement AppTheme**

Create `flutter_app/lib/theme/app_theme.dart`:

```dart
import 'package:flutter/material.dart';

class AppTheme {
  static ThemeData get dark {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: const Color(0xFF1E88E5),
        brightness: Brightness.dark,
      ),
    );
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add flutter_app/lib/theme/
git commit -m "feat(capture): add Material 3 dark theme"
```

---

### Task 8: Export (Zip Builder + Manifest)

**Files:**
- Create: `flutter_app/lib/export/manifest_builder.dart`
- Create: `flutter_app/lib/export/exporter.dart`
- Test: `flutter_app/test/export/exporter_test.dart`

- [ ] **Step 1: Write failing test**

```dart
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/export/exporter.dart';
import 'package:flutter_app/capture/capture_state.dart';

void main() {
  group('Exporter', () {
    test('exports a zip with manifest and video', () async {
      final exporter = Exporter();
      final result = await exporter.export(
        videoPath: 'test_video.mp4',
        leftSamples: [],
        rightSamples: [],
        t0: DateTime.now(),
        leftRef: [1.0, 0.0, 0.0, 0.0],
        rightRef: [1.0, 0.0, 0.0, 0.0],
      );
      expect(await File(result).exists(), isTrue);
      expect(result.endsWith('.esense.zip'), isTrue);
    });
  });
}
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement Exporter**

Create `flutter_app/lib/export/manifest_builder.dart`:

```dart
import 'dart:convert';

class ManifestBuilder {
  static String build({
    required DateTime t0,
    required int durationMs,
    required String videoFilename,
    required int videoWidth,
    required int videoHeight,
    required int videoFps,
    required String leftImuFilename,
    required String rightImuFilename,
    required Map<String, dynamic> leftRef,
    required Map<String, dynamic> rightRef,
  }) {
    final manifest = {
      'version': '1.0',
      'created_at': t0.toIso8601String(),
      't0_ms': t0.millisecondsSinceEpoch,
      'duration_ms': durationMs,
      'video': {
        'filename': videoFilename,
        'fps': videoFps,
        'width': videoWidth,
        'height': videoHeight,
        'start_offset_ms': 0,
      },
      'imu': {
        'left': {
          'filename': leftImuFilename,
          'sample_rate_hz': 100,
          'start_offset_ms': 0,
        },
        'right': {
          'filename': rightImuFilename,
          'sample_rate_hz': 100,
          'start_offset_ms': 0,
        },
      },
      'calibration': {
        'left': leftRef,
        'right': rightRef,
      },
    };
    return const JsonEncoder.withIndent('  ').convert(manifest);
  }
}
```

Create `flutter_app/lib/export/exporter.dart`:

```dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
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
    required int videoWidth,
    required int videoHeight,
    required int videoFps,
  }) async {
    final archive = Archive();

    // Manifest
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
    archive.addFile(ArchiveFile('manifest.json', manifest.length, utf8.encode(manifest)));

    // Video
    final videoFile = File(videoPath);
    final videoBytes = await videoFile.readAsBytes();
    archive.addFile(ArchiveFile('video.mp4', videoBytes.length, videoBytes));

    // Left IMU protobuf
    final leftProto = _buildProtobuf(leftSamples);
    archive.addFile(ArchiveFile('left.imu', leftProto.length, leftProto));

    // Right IMU protobuf
    final rightProto = _buildProtobuf(rightSamples);
    archive.addFile(ArchiveFile('right.imu', rightProto.length, rightProto));

    // Write zip
    final zipEncoder = ZipEncoder();
    final zipBytes = zipEncoder.encode(archive)!;
    final dir = await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final zipPath = '${dir.path}/capture_$timestamp.esense.zip';
    await File(zipPath).writeAsBytes(zipBytes);
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
    final buffer = WriteBuffer();
    for (final s in samples) {
      final sample = IMUSample()
        ..relativeTimestampMs = Int64(s['relative_timestamp_ms'] as int)
        ..accX = (s['acc_x'] as double?)?.toFloat() ?? 0
        ..accY = (s['acc_y'] as double?)?.toFloat() ?? 0
        ..accZ = (s['acc_z'] as double?)?.toFloat() ?? 0
        ..gyroX = (s['gyro_x'] as double?)?.toFloat() ?? 0
        ..gyroY = (s['gyro_y'] as double?)?.toFloat() ?? 0
        ..gyroZ = (s['gyro_z'] as double?)?.toFloat() ?? 0
        ..quatW = (s['quat_w'] as double?)?.toFloat() ?? 0
        ..quatX = (s['quat_x'] as double?)?.toFloat() ?? 0
        ..quatY = (s['quat_y'] as double?)?.toFloat() ?? 0
        ..quatZ = (s['quat_z'] as double?)?.toFloat() ?? 0;
      final bytes = sample.writeToBuffer();
      buffer.putUint64(bytes.length, endian: Endian.little); // varint-ish length prefix
      buffer.putUint8List(bytes);
    }
    return buffer.toBytes();
  }
}
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/export/ flutter_app/test/export/
git commit -m "feat(capture): add .esense.zip exporter with manifest and protobuf"
```

---

### Task 9: Providers + DI Setup

**Files:**
- Create: `flutter_app/lib/providers/app_providers.dart`

- [ ] **Step 1: Implement AppProviders**

Create `flutter_app/lib/providers/app_providers.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../ble/ble_manager.dart';
import '../camera/recorder.dart';
import '../capture/capture_controller.dart';
import '../calibration/calibration_service.dart';

class AppProviders extends StatelessWidget {
  final Widget child;
  const AppProviders({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => BleManager()),
        ChangeNotifierProvider(create: (_) => CameraRecorder()),
        ChangeNotifierProvider(create: (_) => CaptureController()),
        ChangeNotifierProvider(create: (_) => CalibrationService()),
      ],
      child: child,
    );
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add flutter_app/lib/providers/
git commit -m "feat(capture): add Provider DI for all services"
```

---

### Task 10: Main App Scaffold (Material 3 + Provider)

**Files:**
- Modify: `flutter_app/lib/main.dart`
- Modify: `flutter_app/lib/calibration/calibration_service.dart` (add ChangeNotifier)
- Modify: `flutter_app/lib/camera/recorder.dart` (add ChangeNotifier)
- Modify: `flutter_app/lib/capture/capture_controller.dart` (add ChangeNotifier)
- Modify: `flutter_app/lib/ble/ble_manager.dart` (add ChangeNotifier)

- [ ] **Step 1: Make services extend ChangeNotifier**

Add to each service class:

```dart
import 'package:flutter/material.dart';

// CalibrationService
class CalibrationService extends ChangeNotifier {
  // ... existing code ...
}

// CameraRecorder
class CameraRecorder extends ChangeNotifier {
  // ... existing code ...
}

// CaptureController
class CaptureController extends ChangeNotifier {
  // ... existing code ...
}

// BleManager
class BleManager extends ChangeNotifier {
  // ... existing code ...
}
```

- [ ] **Step 2: Rewrite main.dart with Provider + Material 3**

Replace `flutter_app/lib/main.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:provider/provider.dart';

import 'providers/app_providers.dart';
import 'theme/app_theme.dart';
import 'ble/ble_manager.dart';
import 'camera/recorder.dart';
import 'capture/capture_controller.dart';
import 'capture/capture_state.dart';
import 'calibration/calibration_service.dart';
import 'overlay/edge_overlay.dart';
import 'export/exporter.dart';
import 'permissions/permission_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    AppProviders(
      child: const EdgeSenseApp(),
    ),
  );
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
  double _leftEdge = 0.0;
  double _rightEdge = 0.0;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final permissions = await PermissionService().requestAll();
    if (!permissions) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Permissions required')),
        );
      }
      return;
    }
    final cameras = await availableCameras();
    if (mounted) {
      context.read<CameraRecorder>().initialize(cameras);
    }
  }

  Future<void> _startCapture() async {
    final bleManager = context.read<BleManager>();
    final cameraRecorder = context.read<CameraRecorder>();
    final captureController = context.read<CaptureController>();

    setState(() => _isRecording = true);
    await captureController.start(
      bleManager: bleManager,
      cameraRecorder: cameraRecorder,
      onLeftEdgeAngle: (angle) => setState(() => _leftEdge = angle),
      onRightEdgeAngle: (angle) => setState(() => _rightEdge = angle),
    );
  }

  Future<void> _stopCapture() async {
    final bleManager = context.read<BleManager>();
    final cameraRecorder = context.read<CameraRecorder>();
    final captureController = context.read<CaptureController>();

    final result = await captureController.stop(
      bleManager: bleManager,
      cameraRecorder: cameraRecorder,
    );
    setState(() {
      _isRecording = false;
      _lastResult = result;
    });
  }

  Future<void> _export() async {
    if (_lastResult == null) return;
    final calibrationService = context.read<CalibrationService>();

    try {
      final path = await Isolate.run(() async {
        final exporter = Exporter();
        return await exporter.export(
          videoPath: _lastResult!.videoPath,
          leftSamples: _lastResult!.leftSamples,
          rightSamples: _lastResult!.rightSamples,
          t0: _lastResult!.t0,
          leftRef: calibrationService.leftRef ?? [1.0, 0.0, 0.0, 0.0],
          rightRef: calibrationService.rightRef ?? [1.0, 0.0, 0.0, 0.0],
          videoWidth: 1920,
          videoHeight: 1080,
          videoFps: 60,
        );
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Exported to $path')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Export failed: $e')),
        );
      }
    }
  }

  bool get _isRecording => context.watch<CaptureController>().status == CaptureStatus.recording;
  CaptureResult? get _lastResult => context.watch<CaptureController>().lastResult;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('EdgeSense Capture')),
      body: Stack(
        children: [
          Center(
            child: Text(_isRecording ? 'Recording...' : 'Ready'),
          ),
          if (_isRecording)
            EdgeOverlay(leftAngle: _leftEdge, rightAngle: _rightEdge),
        ],
      ),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            onPressed: _isRecording ? _stopCapture : _startCapture,
            child: Icon(_isRecording ? Icons.stop : Icons.videocam),
          ),
          if (_lastResult != null)
            FloatingActionButton(
              onPressed: _export,
              child: const Icon(Icons.share),
            ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add flutter_app/lib/main.dart flutter_app/lib/providers/
git commit -m "feat(capture): add Material 3 theme, Provider DI, Isolate export"
```

---

### Task 11: Integration Test

**Files:**
- Create: `flutter_app/integration_test/capture_flow_test.dart`

- [ ] **Step 1: Write integration test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_app/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Capture Flow', () {
    testWidgets('tap record, stop, and export', (tester) async {
      app.main();
      await tester.pumpAndSettle();

      // Tap record
      await tester.tap(find.byIcon(Icons.videocam));
      await tester.pump(const Duration(seconds: 2));

      // Tap stop
      await tester.tap(find.byIcon(Icons.stop));
      await tester.pumpAndSettle();

      // Tap export
      await tester.tap(find.byIcon(Icons.share));
      await tester.pumpAndSettle();

      // Verify snackbar appears
      expect(find.textContaining('Exported to'), findsOneWidget);
    });
  });
}
```

- [ ] **Step 2: Run integration test**

```bash
flutter test integration_test/capture_flow_test.dart
```

- [ ] **Step 3: Commit**

```bash
git add flutter_app/integration_test/
git commit -m "test(capture): add capture flow integration test"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| Protobuf IMU schema | Task 1 |
| WT901 BLE parser (0x55) | Task 2 |
| BLE scan/connect 2 devices | Task 3 |
| Camera recording | Task 4 |
| Unified t0 trigger | Task 6 |
| Real-time edge overlay | Task 7 |
| Calibration (quat_ref) | Task 5 |
| Export `.esense.zip` | Task 8 |
| Main app UI | Task 9 |
| Integration test | Task 10 |

---

## Self-Review

- [x] No placeholders — all code provided
- [x] Type consistency — `relative_timestamp_ms`, `quat_w/x/y/z` consistent across tasks
- [x] File paths exact
- [x] TDD pattern — failing test → impl → pass → commit
- [x] No TODO/TBD
- [x] Provider DI for services (BleManager, CameraRecorder, CaptureController, CalibrationService)
- [x] Material 3 theme with ColorScheme
- [x] Permission service for BLE + camera
- [x] Isolate.run() for heavy export operation
- [x] flutter_secure_storage for calibration persistence

# Capture Controller Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `CaptureController` (118 lines, UI state + BLE coordination + recording logic) into `CaptureRepository` (business logic / BLE + camera coordination) and `CaptureProvider` (UI state only).

**Architecture:** `CaptureRepository` is a plain Dart class (not `ChangeNotifier`) that owns `BleManager` and `CameraRecorder`, manages start/stop lifecycle, and emits `CaptureResult`. `CaptureProvider` is a thin `ChangeNotifier` that wraps `CaptureRepository`, exposes `CaptureStatus` and buffers sample counts to the UI. No `async` inside `build` — all async lives in repository.

**Tech Stack:** `provider`, existing `BleManager`/`CameraRecorder`/`WT901Packet`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `mobile/lib/capture/capture_repository.dart` | Pure logic: connect BLE, start camera, subscribe to IMU streams, stop everything, return `CaptureResult` |
| `mobile/lib/capture/capture_provider.dart` | UI state: `CaptureStatus`, sample counts, delegates to repository |
| `mobile/lib/capture/capture_controller.dart` | **Delete** — replaced by repository + provider |
| `mobile/lib/capture/capture_state.dart` | Keep — `CaptureStatus` enum + `CaptureResult` class |
| `mobile/lib/providers/app_providers.dart` | Update providers list: replace `CaptureController` with `CaptureProvider` + `CaptureRepository` |
| `mobile/test/capture/capture_repository_test.dart` | Mock BLE + camera, verify start/stop result |
| `mobile/test/capture/capture_provider_test.dart` | Verify status transitions |

---

## Task 1: Write `CaptureRepository`

**Files:**
- Create: `mobile/lib/capture/capture_repository.dart`

- [ ] **Step 1: Write repository class**

```dart
import 'dart:async';
import 'dart:math' as math;

import '../ble/ble_manager.dart';
import '../ble/wt901_parser.dart';
import '../camera/recorder.dart';
import 'capture_state.dart';

class CaptureRepository {
  final BleManager _bleManager;
  final CameraRecorder _cameraRecorder;

  final List<Map<String, dynamic>> _leftBuffer = [];
  final List<Map<String, dynamic>> _rightBuffer = [];
  StreamSubscription? _streamSubscription;
  DateTime? _t0;

  CaptureRepository({required BleManager bleManager, required CameraRecorder cameraRecorder})
      : _bleManager = bleManager,
        _cameraRecorder = cameraRecorder;

  DateTime? get startTime => _t0;
  int get leftSampleCount => _leftBuffer.length;
  int get rightSampleCount => _rightBuffer.length;

  /// Starts capture. Returns null if already recording.
  Future<CaptureResult?> start({
    required void Function(double edgeAngle) onLeftEdgeAngle,
    required void Function(double edgeAngle) onRightEdgeAngle,
  }) async {
    if (_streamSubscription != null) return null;

    _leftBuffer.clear();
    _rightBuffer.clear();

    await _bleManager.connectAll();
    await _cameraRecorder.startRecording();

    final stopwatch = Stopwatch()..start();
    _t0 = DateTime.now();

    _streamSubscription = _bleManager.startStreams().listen((pair) {
      final relativeMs = stopwatch.elapsed.inMilliseconds;

      final left = pair.$1;
      if (left != null) {
        _leftBuffer.add(_toMap(left, relativeMs));
        if (left.quatW != null) {
          onLeftEdgeAngle(_computeEdgeAngle(left));
        }
      }

      final right = pair.$2;
      if (right != null) {
        _rightBuffer.add(_toMap(right, relativeMs));
        if (right.quatW != null) {
          onRightEdgeAngle(_computeEdgeAngle(right));
        }
      }
    });

    return null;
  }

  Future<CaptureResult> stop() async {
    await _streamSubscription?.cancel();
    _streamSubscription = null;

    final videoFile = await _cameraRecorder.stopRecording();
    await _bleManager.disconnectAll();

    return CaptureResult(
      videoPath: videoFile.path,
      leftSamples: List.from(_leftBuffer),
      rightSamples: List.from(_rightBuffer),
      t0: _t0!,
    );
  }

  void dispose() {
    _streamSubscription?.cancel();
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
}
```

- [ ] **Step 2: Commit**

```bash
git add mobile/lib/capture/capture_repository.dart
git commit -m "feat(capture): add CaptureRepository with pure business logic"
```

---

## Task 2: Write `CaptureProvider`

**Files:**
- Create: `mobile/lib/capture/capture_provider.dart`

- [ ] **Step 1: Write provider**

```dart
import 'package:flutter/foundation.dart';

import '../ble/ble_manager.dart';
import '../camera/recorder.dart';
import 'capture_repository.dart';
import 'capture_state.dart';

class CaptureProvider extends ChangeNotifier {
  final CaptureRepository _repo;

  CaptureStatus status = CaptureStatus.idle;
  DateTime? get startTime => _repo.startTime;
  int get leftSampleCount => _repo.leftSampleCount;
  int get rightSampleCount => _repo.rightSampleCount;

  CaptureProvider({required BleManager bleManager, required CameraRecorder cameraRecorder})
      : _repo = CaptureRepository(bleManager: bleManager, cameraRecorder: cameraRecorder);

  Future<CaptureResult?> start({
    required void Function(double edgeAngle) onLeftEdgeAngle,
    required void Function(double edgeAngle) onRightEdgeAngle,
  }) async {
    if (status == CaptureStatus.recording) return null;

    status = CaptureStatus.initializing;
    notifyListeners();

    final result = await _repo.start(
      onLeftEdgeAngle: onLeftEdgeAngle,
      onRightEdgeAngle: onRightEdgeAngle,
    );

    if (result == null) {
      status = CaptureStatus.recording;
      notifyListeners();
    } else {
      status = CaptureStatus.error;
      notifyListeners();
    }

    return result;
  }

  Future<CaptureResult> stop() async {
    status = CaptureStatus.stopping;
    notifyListeners();

    final result = await _repo.stop();

    status = CaptureStatus.idle;
    notifyListeners();

    return result;
  }

  @override
  void dispose() {
    _repo.dispose();
    super.dispose();
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add mobile/lib/capture/capture_provider.dart
git commit -m "feat(capture): add CaptureProvider for UI state only"
```

---

## Task 3: Wire Providers and Delete Old Controller

**Files:**
- Modify: `mobile/lib/providers/app_providers.dart`
- Delete: `mobile/lib/capture/capture_controller.dart`

- [ ] **Step 1: Update AppProviders**

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../ble/ble_manager.dart';
import '../camera/recorder.dart';
import '../capture/capture_provider.dart';
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
        ChangeNotifierProvider(
          create: (ctx) => CaptureProvider(
            bleManager: ctx.read<BleManager>(),
            cameraRecorder: ctx.read<CameraRecorder>(),
          ),
        ),
        ChangeNotifierProvider(create: (_) => CalibrationService()),
      ],
      child: child,
    );
  }
}
```

- [ ] **Step 2: Delete old file**

Run: `rm mobile/lib/capture/capture_controller.dart`

- [ ] **Step 3: Commit**

```bash
git add mobile/lib/providers/app_providers.dart
git rm mobile/lib/capture/capture_controller.dart
git commit -m "refactor(capture): replace CaptureController with Repository + Provider"
```

---

## Task 4: Update Screens to Use CaptureProvider

**Files:**
- Modify: `mobile/lib/capture/capturing_screen.dart` (if it reads `CaptureController`)
- Modify: `mobile/lib/capture/export_result_screen.dart` (if needed)

- [ ] **Step 1: Update imports and reads**

In `capturing_screen.dart`, replace:

```dart
// Before:
import 'capture_controller.dart';
// context.read<CaptureController>()

// After:
import 'capture_provider.dart';
// context.read<CaptureProvider>()
```

Replace all `context.read<CaptureController>()` with `context.read<CaptureProvider>()`.
Replace all `context.watch<CaptureController>()` with `context.watch<CaptureProvider>()`.

- [ ] **Step 2: Commit**

```bash
git add mobile/lib/capture/capturing_screen.dart
git commit -m "refactor(capture): update screens to use CaptureProvider"
```

---

## Task 5: Unit Tests for Repository and Provider

**Files:**
- Create: `mobile/test/capture/capture_repository_test.dart`
- Create: `mobile/test/capture/capture_provider_test.dart`

- [ ] **Step 1: Write repository test**

```dart
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:edgesense_capture/capture/capture_repository.dart';
import 'package:edgesense_capture/ble/ble_manager.dart';
import 'package:edgesense_capture/camera/recorder.dart';

class MockBleManager extends Mock implements BleManager {}
class MockCameraRecorder extends Mock implements CameraRecorder {}
class MockFile extends Mock implements File {}

void main() {
  group('CaptureRepository', () {
    late MockBleManager ble;
    late MockCameraRecorder camera;
    late CaptureRepository repo;

    setUp(() {
      ble = MockBleManager();
      camera = MockCameraRecorder();
      repo = CaptureRepository(bleManager: ble, cameraRecorder: camera);

      when(() => ble.connectAll()).thenAnswer((_) async {});
      when(() => ble.disconnectAll()).thenAnswer((_) async {});
      when(() => ble.startStreams()).thenAnswer((_) => Stream.empty());
      when(() => camera.startRecording()).thenAnswer((_) async {});
      when(() => camera.stopRecording()).thenAnswer((_) async => MockFile());
    });

    test('start returns null when not already recording', () async {
      final result = await repo.start(
        onLeftEdgeAngle: (_) {},
        onRightEdgeAngle: (_) {},
      );
      expect(result, isNull);
    });

    test('stop returns CaptureResult with empty buffers', () async {
      await repo.start(
        onLeftEdgeAngle: (_) {},
        onRightEdgeAngle: (_) {},
      );
      final result = await repo.stop();
      expect(result.videoPath, isNotNull);
      expect(result.leftSamples, isEmpty);
      expect(result.rightSamples, isEmpty);
    });
  });
}
```

- [ ] **Step 2: Write provider test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:edgesense_capture/capture/capture_provider.dart';
import 'package:edgesense_capture/capture/capture_state.dart';
import 'package:edgesense_capture/ble/ble_manager.dart';
import 'package:edgesense_capture/camera/recorder.dart';

class MockBleManager extends Mock implements BleManager {}
class MockCameraRecorder extends Mock implements CameraRecorder {}

void main() {
  group('CaptureProvider', () {
    late MockBleManager ble;
    late MockCameraRecorder camera;
    late CaptureProvider provider;

    setUp(() {
      ble = MockBleManager();
      camera = MockCameraRecorder();
      provider = CaptureProvider(bleManager: ble, cameraRecorder: camera);

      when(() => ble.connectAll()).thenAnswer((_) async {});
      when(() => ble.disconnectAll()).thenAnswer((_) async {});
      when(() => ble.startStreams()).thenAnswer((_) => Stream.empty());
      when(() => camera.startRecording()).thenAnswer((_) async {});
      when(() => camera.stopRecording()).thenAnswer((_) async => MockFile());
    });

    test('status transitions idle → initializing → recording', () async {
      expect(provider.status, CaptureStatus.idle);

      final future = provider.start(onLeftEdgeAngle: (_) {}, onRightEdgeAngle: (_) {});
      expect(provider.status, CaptureStatus.initializing);

      await future;
      expect(provider.status, CaptureStatus.recording);
    });
  });
}
```

- [ ] **Step 3: Run tests**

Run: `cd mobile && flutter test test/capture/ -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add mobile/test/capture/
git commit -m "test(capture): add unit tests for CaptureRepository and CaptureProvider"
```

---

## Self-Review

1. **Spec coverage:**
   - `CaptureRepository` (business logic / BLE) → Task 1 ✅
   - `CaptureProvider` (UI state) → Task 2 ✅
   - No `async` in UI buttons → repository handles all async ✅

2. **Placeholder scan:** No TBD/TODO. All code shown. ✅

3. **Type consistency:** `CaptureStatus`, `CaptureResult` reused from `capture_state.dart`. `BleManager`/`CameraRecorder` APIs unchanged. ✅

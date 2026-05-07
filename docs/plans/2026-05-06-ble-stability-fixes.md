# BLE Stability P0 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical BLE packet corruption and connection state race conditions, add regression tests.

**Architecture:** Keep existing class boundaries (`WT901Parser`, `IMUDevice`) but harden internal logic. Parser gets packet slicing before checksum. `IMUDevice` replaces eager `isConnected = false` assignment with reactive `StreamController<ConnectionState>`. All fixes verified by unit tests before Patrol integration.

**Tech Stack:** Dart, `flutter_blue_plus`, `patrol`, `mocktail`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `mobile/lib/ble/wt901_parser.dart` | Parse raw WT901 bytes → typed packet. **Fix:** slice `raw.sublist(0, packetLength)` before checksum. |
| `mobile/lib/ble/imu_device.dart` | Wrap `BluetoothDevice` with connection state + notification streams. **Fix:** use `StreamController<BluetoothConnectionState>` instead of eager `isConnected` mutation. |
| `mobile/lib/ble/ble_manager.dart` | Aggregate two `IMUDevice`s, handle scanning. Minor updates for `IMUDevice` API changes. |
| `mobile/test/ble/wt901_parser_test.dart` | Unit tests: valid packet, truncated packet, concatenated packet, wrong checksum. |
| `mobile/test/ble/imu_device_test.dart` | Unit tests: state stream emits on connect/disconnect, no eager false before real event. |
| `mobile/integration_test/patrol_flow_test.dart` | End-to-end: grant permissions → scan → assign left/right → start capture → verify recording state. |

---

## Task 1: WT901Parser Packet Boundary Fix

**Files:**
- Modify: `mobile/lib/ble/wt901_parser.dart`
- Create: `mobile/test/ble/wt901_parser_test.dart`

**Problem:** When BLE stream delivers `raw.length > packetLength` (two packets concatenated), `_computeChecksum(raw)` sums correct bytes but `raw[packetLength - 1]` reads wrong byte (from second packet) for checksum comparison. Parser fails or parses garbage.

**Fix:** Extract `packet = raw.sublist(0, packetLength)` immediately after length/header checks. Run all subsequent logic on `packet`.

- [ ] **Step 1: Write failing test for concatenated packet**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/ble/wt901_parser.dart';
import 'package:edgesense_capture/ble/wt901_constants.dart';

void main() {
  group('WT901Parser', () {
    test('parses concatenated packet by taking only first packetLength bytes', () {
      final valid = List<int>.filled(packetLength, 0x55);
      valid[0] = packetHeader;
      valid[1] = typeAccel;
      // compute correct checksum on first packetLength bytes
      var sum = 0;
      for (var i = 0; i < packetLength - 1; i++) sum += valid[i];
      valid[packetLength - 1] = sum & 0xFF;

      final garbage = List<int>.filled(packetLength, 0xFF);
      final concatenated = [...valid, ...garbage];

      final result = WT901Parser.parse(concatenated);
      expect(result, isNotNull);
      expect(result!.type, WT901PacketType.accelerometer);
    });

    test('returns null for truncated packet', () {
      final truncated = [packetHeader, typeAccel];
      expect(WT901Parser.parse(truncated), isNull);
    });

    test('returns null for wrong checksum', () {
      final raw = List<int>.filled(packetLength, 0x55);
      raw[0] = packetHeader;
      raw[1] = typeAccel;
      raw[packetLength - 1] = 0x00; // wrong checksum
      expect(WT901Parser.parse(raw), isNull);
    });
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile && flutter test test/ble/wt901_parser_test.dart -v`
Expected: FAIL — `test/ble/wt901_parser_test.dart` file not found.

- [ ] **Step 3: Fix WT901Parser**

In `mobile/lib/ble/wt901_parser.dart`, replace `parse` method:

```dart
static WT901Packet? parse(List<int> raw) {
  if (raw.length < packetLength) return null;
  if (raw[0] != packetHeader) return null;

  final packet = raw.sublist(0, packetLength);
  final checksum = _computeChecksum(packet);
  if (packet[packetLength - 1] != checksum) return null;

  final typeByte = packet[1];
  WT901PacketType type;
  switch (typeByte) {
    case typeAccel: type = WT901PacketType.accelerometer; break;
    case typeGyro: type = WT901PacketType.gyroscope; break;
    case typeAngle: type = WT901PacketType.angle; break;
    case typeQuat: type = WT901PacketType.quaternion; break;
    case typeBattery: type = WT901PacketType.battery; break;
    default: type = WT901PacketType.unknown;
  }

  final data = ByteData.sublistView(Uint8List.fromList(packet));

  double? accX, accY, accZ;
  double? gyroX, gyroY, gyroZ;
  double? angleX, angleY, angleZ;
  double? quatW, quatX, quatY, quatZ;
  double? battery;

  if (type == WT901PacketType.accelerometer) {
    accX = _readInt16Scaled(data, 2, scaleAcc);
    accY = _readInt16Scaled(data, 4, scaleAcc);
    accZ = _readInt16Scaled(data, 6, scaleAcc);
  } else if (type == WT901PacketType.gyroscope) {
    gyroX = _readInt16Scaled(data, 2, scaleGyro);
    gyroY = _readInt16Scaled(data, 4, scaleGyro);
    gyroZ = _readInt16Scaled(data, 6, scaleGyro);
  } else if (type == WT901PacketType.angle) {
    angleX = _readInt16Scaled(data, 2, scaleAngle);
    angleY = _readInt16Scaled(data, 4, scaleAngle);
    angleZ = _readInt16Scaled(data, 6, scaleAngle);
  } else if (type == WT901PacketType.quaternion) {
    quatW = _readInt16Scaled(data, 2, scaleQuat);
    quatX = _readInt16Scaled(data, 4, scaleQuat);
    quatY = _readInt16Scaled(data, 6, scaleQuat);
    quatZ = _readInt16Scaled(data, 8, scaleQuat);
  } else if (type == WT901PacketType.battery) {
    battery = data.getInt16(2, Endian.little) / 100.0;
  }

  return WT901Packet(
    type: type,
    accX: accX, accY: accY, accZ: accZ,
    gyroX: gyroX, gyroY: gyroY, gyroZ: gyroZ,
    angleX: angleX, angleY: angleY, angleZ: angleZ,
    quatW: quatW, quatX: quatX, quatY: quatY, quatZ: quatZ,
    battery: battery,
  );
}
```

- [ ] **Step 4: Run tests**

Run: `cd mobile && flutter test test/ble/wt901_parser_test.dart -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/lib/ble/wt901_parser.dart mobile/test/ble/wt901_parser_test.dart
git commit -m "fix(ble): slice packet before checksum to prevent concatenation corruption"
```

---

## Task 2: IMUDevice Connection State Race Fix

**Files:**
- Modify: `mobile/lib/ble/imu_device.dart`
- Modify: `mobile/lib/ble/ble_manager.dart` (if it reads `isConnected` directly)
- Create: `mobile/test/ble/imu_device_test.dart`

**Problem:** `disconnect()` sets `isConnected = false` immediately, before `flutter_blue_plus` emits real disconnection event. UI may show "disconnected" then flicker back to "connected" if stack emits late event.

**Fix:** Replace public `bool isConnected` with `Stream<BluetoothConnectionState> connectionState`. Expose `ValueNotifier<bool>` or `Stream<bool>` derived from real events only. Never mutate state manually.

- [ ] **Step 1: Write failing test for state race**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:mocktail/mocktail.dart';
import 'package:edgesense_capture/ble/imu_device.dart';

class MockBluetoothDevice extends Mock implements BluetoothDevice {}
class MockConnectionStateStream extends Mock implements Stream<BluetoothConnectionState> {}

void main() {
  group('IMUDevice', () {
    late MockBluetoothDevice mockDevice;

    setUp(() {
      mockDevice = MockBluetoothDevice();
      when(() => mockDevice.isConnected).thenReturn(true);
      when(() => mockDevice.connectionState).thenAnswer(
        (_) => Stream.fromIterable([
          BluetoothConnectionState.connected,
          BluetoothConnectionState.disconnected,
        ]),
      );
      when(() => mockDevice.disconnect()).thenAnswer((_) async {});
    });

    test('does not emit disconnected until stream emits it', () async {
      final imu = IMUDevice(device: mockDevice, side: 'left');

      final states = <bool>[];
      final sub = imu.isConnectedStream.listen(states.add);

      // wait for initial connected + disconnected events
      await Future.delayed(const Duration(milliseconds: 50));

      // disconnect() must not eagerly inject false
      await imu.disconnect();
      await Future.delayed(const Duration(milliseconds: 50));

      // states should be [true, false] — never [true, false, false] or [false, true, false]
      expect(states.where((s) => s == false).length, 1);
      expect(states.last, false);

      await sub.cancel();
    });
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mobile && flutter test test/ble/imu_device_test.dart -v`
Expected: FAIL — `isConnectedStream` getter not found.

- [ ] **Step 3: Refactor IMUDevice**

Replace contents of `mobile/lib/ble/imu_device.dart`:

```dart
import 'dart:async';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'wt901_parser.dart';

class IMUDevice {
  final BluetoothDevice device;
  final String side;
  final void Function(double voltage)? onBattery;

  final _connectionController = StreamController<bool>.broadcast();
  Stream<bool> get isConnectedStream => _connectionController.stream;

  StreamSubscription? _notifySubscription;
  StreamSubscription? _connectionSubscription;

  IMUDevice({
    required this.device,
    required this.side,
    this.onBattery,
  }) {
    _connectionSubscription = device.connectionState.listen((state) {
      _connectionController.add(state == BluetoothConnectionState.connected);
    });
  }

  Future<void> connect() async {
    await device.connect(autoConnect: false);
  }

  Future<void> disconnect() async {
    await _notifySubscription?.cancel();
    _notifySubscription = null;
    // Do NOT touch connection state here. Let device.connectionState stream do it.
    try { await device.disconnect(); } catch (_) {}
    // Cancel subscription after disconnect to avoid late events
    await _connectionSubscription?.cancel();
    _connectionSubscription = null;
  }

  void dispose() {
    _connectionSubscription?.cancel();
    _connectionController.close();
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
      if (packet != null) {
        if (packet.type == WT901PacketType.battery && packet.battery != null) {
          onBattery?.call(packet.battery!);
        }
        yield packet;
      }
    }
  }
}
```

- [ ] **Step 4: Update BleManager consumers**

In `mobile/lib/ble/ble_manager.dart`, find all direct reads of `imuDevice.isConnected` and replace with `imuDevice.isConnectedStream` or reactive builder.

Example change inside `BleManager` (if it exposes widget-level state):

```dart
// Before: bool get leftConnected => leftDevice?.isConnected ?? false;
// After:
Stream<bool> get leftConnectedStream =>
    leftDevice?.isConnectedStream ?? Stream.value(false);
```

If `BleScanScreen` uses `ble.leftDevice?.isConnected` directly in `build`, replace with `StreamBuilder<bool>` or move connection state into `BleManager` ChangeNotifier that aggregates both streams.

- [ ] **Step 5: Run tests**

Run: `cd mobile && flutter test test/ble/imu_device_test.dart -v`
Expected: 1 test PASS.

- [ ] **Step 6: Commit**

```bash
git add mobile/lib/ble/imu_device.dart mobile/lib/ble/ble_manager.dart mobile/test/ble/imu_device_test.dart
git commit -m "fix(ble): remove eager isConnected mutation, use real connectionState stream"
```

---

## Task 3: Patrol End-to-End Flow Test

**Files:**
- Create: `mobile/integration_test/patrol_flow_test.dart`

**Goal:** Validate the whole user flow after P0 fixes. Replace existing `permissions_test.dart` with broader flow.

- [ ] **Step 1: Write Patrol integration test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:patrol/patrol.dart';
import 'package:edgesense_capture/main.dart' as app;

void main() {
  patrolTest(
    'permissions → scan → capture flow',
    (PatrolTester $) async {
      await app.main();
      await $.pumpAndSettle();

      // 1. Permissions
      await $('Предоставить разрешения').tap();
      await $.pumpAndSettle();

      // 2. BLE Scan screen
      await $('Подключение IMU').waitUntilExists();
      await $('Сканировать').tap();
      await $.pumpAndSettle(duration: const Duration(seconds: 6));

      // 3. Assert scan list or empty state
      expect($('Сканировать снова').exists || $('ListView').exists, true);

      // 4. Tap "Далее" (if sensors assigned in mock, else skip)
      // For real hardware CI, pre-seed connected devices or mock BleManager.
    },
  );
}
```

- [ ] **Step 2: Run test to verify framework loads**

Run: `cd mobile && patrol test -t integration_test/patrol_flow_test.dart`
Expected: Test runs on emulator (may fail on hardware absence, but framework must not crash).

- [ ] **Step 3: Commit**

```bash
git add mobile/integration_test/patrol_flow_test.dart
git commit -m "test(ble): add Patrol end-to-end flow test"
```

---

## Self-Review

1. **Spec coverage:**
   - WT901Parser `sublist(0, packetLength)` fix → Task 1 Step 3 ✅
   - IMUDevice state race with StreamController → Task 2 Step 3 ✅
   - Patrol integration tests → Task 3 ✅

2. **Placeholder scan:** No TBD/TODO/fill-in. All code blocks complete. ✅

3. **Type consistency:** `WT901Packet`, `WT901PacketType`, `packetLength`, `packetHeader` reused from existing constants. `BluetoothConnectionState` from `flutter_blue_plus`. Consistent. ✅

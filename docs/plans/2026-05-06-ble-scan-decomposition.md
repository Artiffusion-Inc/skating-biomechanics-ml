# BLE Scan Screen Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `ble_scan_screen.dart` (396 lines, 3 widgets + 3 dialogs) into atomic, testable dumb widgets under 150 lines each.

**Architecture:** Extract `_DeviceTile` → `ScannerTile`, `_showAssignSheet` → `ConnectionSheet`, `_showSensorSettings` + `_showRenameDialog` → `DeviceSettingsSheet`. All extracted widgets are `StatelessWidget` with data + callbacks only. `BleScanScreen` becomes a thin scaffold that wires `BleManager` to children.

**Tech Stack:** Flutter, `provider`, existing `BleManager` API

---

## File Structure

| File | Responsibility | Target Lines |
|------|----------------|-------------|
| `mobile/lib/ble/ui/scanner_tile.dart` | Device row: name, battery chip, assignment chip, settings button | < 100 |
| `mobile/lib/ble/ui/connection_sheet.dart` | Bottom sheet for assigning left/right | < 80 |
| `mobile/lib/ble/ui/device_settings_sheet.dart` | Battery, return rate, rename actions | < 120 |
| `mobile/lib/ble/ui/status_bar.dart` | Connection summary banner | < 50 |
| `mobile/lib/ble/ble_scan_screen.dart` | Scaffold + ListView builder only | < 100 |
| `mobile/test/ble/ui/scanner_tile_test.dart` | Widget test for tile rendering | — |

---

## Task 1: Extract `StatusBar`

**Files:**
- Create: `mobile/lib/ble/ui/status_bar.dart`
- Modify: `mobile/lib/ble/ble_scan_screen.dart`

- [ ] **Step 1: Write StatusBar widget**

```dart
import 'package:flutter/material.dart';
import '../../ble/imu_device.dart';

class StatusBar extends StatelessWidget {
  final IMUDevice? leftDevice;
  final IMUDevice? rightDevice;

  const StatusBar({super.key, this.leftDevice, this.rightDevice});

  @override
  Widget build(BuildContext context) {
    final parts = <String>[];
    if (leftDevice != null) {
      parts.add('Левый ${leftDevice!.isConnected ? '✓' : '…'}');
    }
    if (rightDevice != null) {
      parts.add('Правый ${rightDevice!.isConnected ? '✓' : '…'}');
    }

    final leftOk = leftDevice?.isConnected ?? false;
    final rightOk = rightDevice?.isConnected ?? false;
    final color = leftOk && rightOk
        ? Colors.green.shade800
        : leftOk || rightOk
            ? Colors.orange.shade800
            : Colors.grey.shade800;

    final text = parts.isEmpty
        ? 'Нажмите на устройство для назначения'
        : leftOk && rightOk
            ? '${parts.join('  ')}  —  оба подключены'
            : leftOk || rightOk
                ? '${parts.join('  ')}  —  можно продолжить'
                : '${parts.join('  ')}  —  подключение…';

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      color: color,
      child: Text(text, style: const TextStyle(fontWeight: FontWeight.bold)),
    );
  }
}
```

- [ ] **Step 2: Replace `_StatusBar` in ble_scan_screen.dart**

Remove `_StatusBar` class from `ble_scan_screen.dart`. Import `ui/status_bar.dart`. Replace usage:

```dart
// Before:
// _StatusBar(ble: ble)
// After:
StatusBar(leftDevice: ble.leftDevice, rightDevice: ble.rightDevice)
```

- [ ] **Step 3: Commit**

```bash
git add mobile/lib/ble/ui/status_bar.dart mobile/lib/ble/ble_scan_screen.dart
git commit -m "refactor(ble): extract StatusBar from ble_scan_screen"
```

---

## Task 2: Extract `ScannerTile`

**Files:**
- Create: `mobile/lib/ble/ui/scanner_tile.dart`
- Modify: `mobile/lib/ble/ble_scan_screen.dart`

- [ ] **Step 1: Write ScannerTile widget**

```dart
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../../ble/imu_device.dart';

class ScannerTile extends StatelessWidget {
  final ScanResult result;
  final IMUDevice? leftDevice;
  final IMUDevice? rightDevice;
  final double? batteryVoltage;
  final VoidCallback onAssign;
  final VoidCallback onSettings;

  const ScannerTile({
    super.key,
    required this.result,
    this.leftDevice,
    this.rightDevice,
    this.batteryVoltage,
    required this.onAssign,
    required this.onSettings,
  });

  @override
  Widget build(BuildContext context) {
    final device = result.device;
    final isLeft = leftDevice?.device.id == device.id;
    final isRight = rightDevice?.device.id == device.id;
    final name = device.platformName;
    final voltage = batteryVoltage;

    return ListTile(
      leading: Icon(
        Icons.bluetooth,
        color: isLeft || isRight ? Colors.blue : Colors.grey,
      ),
      title: Text(name),
      subtitle: Text(device.id.id),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (voltage != null) _BatteryChip(voltage: voltage),
          if (isLeft)
            _SideChip(label: 'Левый', connected: leftDevice?.isConnected ?? false, color: Colors.blue)
          else if (isRight)
            _SideChip(label: 'Правый', connected: rightDevice?.isConnected ?? false, color: Colors.purple)
          else
            IconButton(
              icon: const Icon(Icons.settings, size: 20),
              onPressed: onSettings,
            ),
        ],
      ),
      onTap: onAssign,
    );
  }
}

class _BatteryChip extends StatelessWidget {
  final double voltage;
  const _BatteryChip({required this.voltage});

  Color get _color => voltage > 3.7
      ? Colors.green
      : voltage > 3.5
          ? Colors.orange
          : Colors.red;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.battery_full, size: 14, color: _color),
          const SizedBox(width: 2),
          Text('${voltage.toStringAsFixed(1)}V', style: TextStyle(fontSize: 11, color: _color)),
        ],
      ),
    );
  }
}

class _SideChip extends StatelessWidget {
  final String label;
  final bool connected;
  final Color color;
  const _SideChip({required this.label, required this.connected, required this.color});

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text('$label ${connected ? '✓' : '…'}'),
      backgroundColor: color.shade800,
    );
  }
}
```

- [ ] **Step 2: Replace `_DeviceTile` in ble_scan_screen.dart**

Remove `_DeviceTile` class. Import `ui/scanner_tile.dart`. Replace `ListView.builder` `itemBuilder`:

```dart
itemBuilder: (ctx, i) => ScannerTile(
  result: devices[i],
  leftDevice: ble.leftDevice,
  rightDevice: ble.rightDevice,
  batteryVoltage: ble.batteryLevels[devices[i].device.id.id],
  onAssign: () => _showAssignSheet(devices[i], ble),
  onSettings: () => _showSensorSettings(devices[i], ble),
),
```

- [ ] **Step 3: Write widget test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:edgesense_capture/ble/ui/scanner_tile.dart';
import 'package:edgesense_capture/ble/imu_device.dart';

class FakeScanResult extends Fake implements ScanResult {
  @override
  BluetoothDevice get device => FakeBluetoothDevice();
}

class FakeBluetoothDevice extends Fake implements BluetoothDevice {
  @override
  DeviceIdentifier get id => DeviceIdentifier('AA:BB:CC:DD:EE:FF');
  @override
  String get platformName => 'WT901';
}

void main() {
  testWidgets('renders device name and battery', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: ScannerTile(
          result: FakeScanResult(),
          batteryVoltage: 3.8,
          onAssign: () {},
          onSettings: () {},
        ),
      ),
    );
    expect(find.text('WT901'), findsOneWidget);
    expect(find.text('3.8V'), findsOneWidget);
  });
}
```

- [ ] **Step 4: Run tests**

Run: `cd mobile && flutter test test/ble/ui/scanner_tile_test.dart -v`
Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/lib/ble/ui/scanner_tile.dart mobile/lib/ble/ble_scan_screen.dart mobile/test/ble/ui/scanner_tile_test.dart
git commit -m "refactor(ble): extract ScannerTile widget with battery chip"
```

---

## Task 3: Extract `ConnectionSheet`

**Files:**
- Create: `mobile/lib/ble/ui/connection_sheet.dart`
- Modify: `mobile/lib/ble/ble_scan_screen.dart`

- [ ] **Step 1: Write ConnectionSheet**

```dart
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

class ConnectionSheet extends StatelessWidget {
  final ScanResult result;
  final bool isLeft;
  final bool isRight;
  final VoidCallback onAssignLeft;
  final VoidCallback onAssignRight;
  final VoidCallback onUnassign;

  const ConnectionSheet({
    super.key,
    required this.result,
    required this.isLeft,
    required this.isRight,
    required this.onAssignLeft,
    required this.onAssignRight,
    required this.onUnassign,
  });

  @override
  Widget build(BuildContext context) {
    if (isLeft || isRight) {
      // Auto-unassign and close immediately
      WidgetsBinding.instance.addPostFrameCallback((_) {
        onUnassign();
        Navigator.pop(context);
      });
      return const SizedBox.shrink();
    }

    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ListTile(
            title: Text(result.device.platformName),
            subtitle: Text(result.device.id.id),
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.skip_previous, color: Colors.blue),
            title: const Text('Левый датчик'),
            onTap: () {
              onAssignLeft();
              Navigator.pop(context);
            },
          ),
          ListTile(
            leading: const Icon(Icons.skip_next, color: Colors.purple),
            title: const Text('Правый датчик'),
            onTap: () {
              onAssignRight();
              Navigator.pop(context);
            },
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 2: Replace `_showAssignSheet` in ble_scan_screen.dart**

Remove `_showAssignSheet` method. Replace tap handler:

```dart
void _showAssignSheet(ScanResult result, BleManager ble) {
  final device = result.device;
  final isLeft = ble.leftDevice?.device.id == device.id;
  final isRight = ble.rightDevice?.device.id == device.id;

  showModalBottomSheet(
    context: context,
    builder: (_) => ConnectionSheet(
      result: result,
      isLeft: isLeft,
      isRight: isRight,
      onAssignLeft: () => ble.assignDevice('left', device),
      onAssignRight: () => ble.assignDevice('right', device),
      onUnassign: () => ble.unassignDevice(isLeft ? 'left' : 'right'),
    ),
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add mobile/lib/ble/ui/connection_sheet.dart mobile/lib/ble/ble_scan_screen.dart
git commit -m "refactor(ble): extract ConnectionSheet bottom sheet"
```

---

## Task 4: Extract `DeviceSettingsSheet`

**Files:**
- Create: `mobile/lib/ble/ui/device_settings_sheet.dart`
- Create: `mobile/lib/ble/ui/rename_dialog.dart`
- Modify: `mobile/lib/ble/ble_scan_screen.dart`

- [ ] **Step 1: Write RenameDialog**

```dart
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../wt901_commander.dart';

class RenameDialog extends StatelessWidget {
  final BluetoothDevice device;
  const RenameDialog({super.key, required this.device});

  @override
  Widget build(BuildContext context) {
    final ctrl = TextEditingController();
    return AlertDialog(
      title: const Text('Переименовать датчик'),
      content: TextField(
        controller: ctrl,
        decoration: const InputDecoration(
          labelText: 'Новый ID (0-255)',
          hintText: 'Например: 1',
        ),
        keyboardType: TextInputType.number,
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Отмена'),
        ),
        FilledButton(
          onPressed: () async {
            final id = int.tryParse(ctrl.text);
            if (id != null && id >= 0 && id <= 255) {
              final commander = WT901Commander(device);
              await commander.rename(id);
              if (context.mounted) Navigator.pop(context);
            }
          },
          child: const Text('Сохранить'),
        ),
      ],
    );
  }
}
```

- [ ] **Step 2: Write DeviceSettingsSheet**

```dart
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../wt901_commander.dart';
import 'rename_dialog.dart';

class DeviceSettingsSheet extends StatelessWidget {
  final ScanResult result;
  final double? batteryVoltage;

  const DeviceSettingsSheet({
    super.key,
    required this.result,
    this.batteryVoltage,
  });

  @override
  Widget build(BuildContext context) {
    final device = result.device;
    final commander = WT901Commander(device);

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(device.platformName,
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            Text(device.id.id, style: const TextStyle(fontSize: 12, color: Colors.white70)),
            const SizedBox(height: 16),
            ListTile(
              leading: Icon(Icons.battery_full, color: _batteryColor(batteryVoltage)),
              title: const Text('Заряд батареи'),
              subtitle: Text(batteryVoltage == null
                  ? 'Неизвестно — запросите'
                  : '${batteryVoltage!.toStringAsFixed(2)} В'),
              trailing: TextButton(
                onPressed: () async {
                  await commander.requestBattery();
                  if (context.mounted) Navigator.pop(context);
                },
                child: const Text('Запросить'),
              ),
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.speed),
              title: const Text('Частота передачи'),
              subtitle: const Text('Гц (после записи переподключить)'),
              trailing: DropdownButton<int>(
                value: null,
                hint: const Text('Выбрать'),
                underline: const SizedBox.shrink(),
                items: const [
                  DropdownMenuItem(value: 0x01, child: Text('0.2 Гц')),
                  DropdownMenuItem(value: 0x02, child: Text('0.5 Гц')),
                  DropdownMenuItem(value: 0x03, child: Text('1 Гц')),
                  DropdownMenuItem(value: 0x04, child: Text('2 Гц')),
                  DropdownMenuItem(value: 0x05, child: Text('5 Гц')),
                  DropdownMenuItem(value: 0x06, child: Text('10 Гц')),
                  DropdownMenuItem(value: 0x07, child: Text('20 Гц')),
                  DropdownMenuItem(value: 0x08, child: Text('50 Гц')),
                  DropdownMenuItem(value: 0x09, child: Text('100 Гц')),
                  DropdownMenuItem(value: 0x0B, child: Text('200 Гц')),
                ],
                onChanged: (code) async {
                  if (code != null) {
                    await commander.setReturnRate(code);
                    if (context.mounted) Navigator.pop(context);
                  }
                },
              ),
            ),
            ListTile(
              leading: const Icon(Icons.edit),
              title: const Text('Переименовать (ID 0-255)'),
              trailing: FilledButton.tonal(
                onPressed: () => showDialog(
                  context: context,
                  builder: (_) => RenameDialog(device: device),
                ),
                child: const Text('Изменить'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _batteryColor(double? v) {
    if (v == null) return Colors.grey;
    if (v > 3.7) return Colors.green;
    if (v > 3.5) return Colors.orange;
    return Colors.red;
  }
}
```

- [ ] **Step 3: Replace `_showSensorSettings` and `_showRenameDialog` in ble_scan_screen.dart**

Remove both methods. Import `ui/device_settings_sheet.dart`. Replace settings tap handler:

```dart
void _showSensorSettings(ScanResult result, BleManager ble) {
  showModalBottomSheet(
    context: context,
    builder: (_) => DeviceSettingsSheet(
      result: result,
      batteryVoltage: ble.batteryLevels[result.device.id.id],
    ),
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add mobile/lib/ble/ui/device_settings_sheet.dart mobile/lib/ble/ui/rename_dialog.dart mobile/lib/ble/ble_scan_screen.dart
git commit -m "refactor(ble): extract DeviceSettingsSheet and RenameDialog"
```

---

## Task 5: Verify Final File Sizes

- [ ] **Step 1: Check line counts**

Run: `wc -l mobile/lib/ble/ble_scan_screen.dart mobile/lib/ble/ui/*.dart`
Expected:
- `ble_scan_screen.dart` < 100 lines
- `status_bar.dart` < 50 lines
- `scanner_tile.dart` < 100 lines
- `connection_sheet.dart` < 80 lines
- `device_settings_sheet.dart` < 120 lines
- `rename_dialog.dart` < 50 lines

- [ ] **Step 2: Run all BLE widget tests**

Run: `cd mobile && flutter test test/ble/ -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git commit -m "test(ble): verify decomposed widgets render correctly"
```

---

## Self-Review

1. **Spec coverage:**
   - `ScannerTile` extracted → Task 2 ✅
   - `ConnectionSheet` extracted → Task 3 ✅
   - `DeviceDialog` (DeviceSettingsSheet + RenameDialog) extracted → Task 4 ✅
   - Goal < 150 lines per file → Task 5 verification ✅

2. **Placeholder scan:** No TBD/TODO. All code shown. ✅

3. **Type consistency:** `ScanResult`, `BleManager`, `WT901Commander` API unchanged. ✅

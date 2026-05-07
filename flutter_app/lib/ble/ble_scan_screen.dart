import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:provider/provider.dart';
import 'ble_manager.dart';
import 'wt901_commander.dart';

class BleScanScreen extends StatefulWidget {
  final VoidCallback onReady;
  const BleScanScreen({super.key, required this.onReady});

  @override
  State<BleScanScreen> createState() => _BleScanScreenState();
}

class _BleScanScreenState extends State<BleScanScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _startScan());
  }

  Future<void> _startScan() async {
    await context.read<BleManager>().startScan();
  }

  @override
  Widget build(BuildContext context) {
    final ble = context.watch<BleManager>();
    final devices = ble.namedScanResults;

    return Scaffold(
      appBar: AppBar(title: const Text('Подключение IMU')),
      body: Column(
        children: [
          _StatusBar(ble: ble),
          if (!ble.isBluetoothOn)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              color: Colors.red.shade900,
              child: const Text('Bluetooth выключен', style: TextStyle(fontSize: 14)),
            ),
          if (!ble.locationPermissionGranted)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              color: Colors.orange.shade900,
              child: const Text(
                'Для BLE сканирования нужно разрешение на местоположение',
                style: TextStyle(fontSize: 14),
              ),
            ),
          if (ble.scanError != null)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              color: Colors.red.shade900,
              child: Text(ble.scanError!, style: const TextStyle(fontSize: 12)),
            ),
          Expanded(
            child: devices.isEmpty
                ? Center(
                    child: ble.isScanning
                        ? const CircularProgressIndicator()
                        : TextButton.icon(
                            onPressed: _startScan,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Сканировать снова'),
                          ),
                  )
                : ListView.builder(
                    itemCount: devices.length,
                    itemBuilder: (ctx, i) => _DeviceTile(
                      result: devices[i],
                      ble: ble,
                      onAssign: () => _showAssignSheet(devices[i], ble),
                      onSettings: () => _showSensorSettings(devices[i], ble),
                    ),
                  ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: ble.isScanning ? null : _startScan,
                    icon: const Icon(Icons.search),
                    label: const Text('Сканировать'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: ble.canProceed ? widget.onReady : null,
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

  void _showAssignSheet(ScanResult result, BleManager ble) {
    final device = result.device;
    final isLeft = ble.leftDevice?.device.id == device.id;
    final isRight = ble.rightDevice?.device.id == device.id;

    if (isLeft) {
      ble.unassignDevice('left');
      return;
    }
    if (isRight) {
      ble.unassignDevice('right');
      return;
    }

    showModalBottomSheet(
      context: context,
      builder: (ctx) => SafeArea(
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

  void _showSensorSettings(ScanResult result, BleManager ble) {
    final voltage = ble.batteryLevels[result.device.id.id];
    final commander = WT901Commander(result.device);

    showModalBottomSheet(
      context: context,
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                result.device.platformName,
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              Text(result.device.id.id, style: const TextStyle(fontSize: 12, color: Colors.white70)),
              const SizedBox(height: 16),
              // Battery display
              ListTile(
                leading: Icon(
                  Icons.battery_full,
                  color: voltage == null
                      ? Colors.grey
                      : voltage > 3.7
                          ? Colors.green
                          : voltage > 3.5
                              ? Colors.orange
                              : Colors.red,
                ),
                title: const Text('Заряд батареи'),
                subtitle: Text(voltage == null ? 'Неизвестно — запросите' : '${voltage.toStringAsFixed(2)} В'),
                trailing: TextButton(
                  onPressed: () async {
                    await commander.requestBattery();
                    if (ctx.mounted) Navigator.pop(ctx);
                  },
                  child: const Text('Запросить'),
                ),
              ),
              const Divider(),
              // Return rate
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
                      if (ctx.mounted) Navigator.pop(ctx);
                    }
                  },
                ),
              ),
              // Rename
              ListTile(
                leading: const Icon(Icons.edit),
                title: const Text('Переименовать (ID 0-255)'),
                trailing: FilledButton.tonal(
                  onPressed: () => _showRenameDialog(result.device),
                  child: const Text('Изменить'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showRenameDialog(BluetoothDevice device) {
    final ctrl = TextEditingController();
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
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
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Отмена'),
          ),
          FilledButton(
            onPressed: () async {
              final id = int.tryParse(ctrl.text);
              if (id != null && id >= 0 && id <= 255) {
                final commander = WT901Commander(device);
                await commander.rename(id);
                if (ctx.mounted) Navigator.pop(ctx);
              }
            },
            child: const Text('Сохранить'),
          ),
        ],
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  final BleManager ble;
  const _StatusBar({required this.ble});

  @override
  Widget build(BuildContext context) {
    final parts = <String>[];
    if (ble.leftDevice != null) {
      parts.add('Левый ${ble.leftDevice!.isConnected ? '✓' : '…'}');
    }
    if (ble.rightDevice != null) {
      parts.add('Правый ${ble.rightDevice!.isConnected ? '✓' : '…'}');
    }

    final leftOk = ble.leftDevice?.isConnected ?? false;
    final rightOk = ble.rightDevice?.isConnected ?? false;
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

class _DeviceTile extends StatelessWidget {
  final ScanResult result;
  final BleManager ble;
  final VoidCallback onAssign;
  final VoidCallback onSettings;

  const _DeviceTile({
    required this.result,
    required this.ble,
    required this.onAssign,
    required this.onSettings,
  });

  @override
  Widget build(BuildContext context) {
    final isLeft = ble.leftDevice?.device.id == result.device.id;
    final isRight = ble.rightDevice?.device.id == result.device.id;
    final name = result.device.platformName;
    final voltage = ble.batteryLevels[result.device.id.id];

    return ListTile(
      leading: Icon(
        Icons.bluetooth,
        color: isLeft || isRight ? Colors.blue : Colors.grey,
      ),
      title: Text(name),
      subtitle: Text(result.device.id.id),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (voltage != null)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    Icons.battery_full,
                    size: 14,
                    color: voltage > 3.7
                        ? Colors.green
                        : voltage > 3.5
                            ? Colors.orange
                            : Colors.red,
                  ),
                  const SizedBox(width: 2),
                  Text(
                    '${voltage.toStringAsFixed(1)}V',
                    style: TextStyle(
                      fontSize: 11,
                      color: voltage > 3.7
                          ? Colors.green
                          : voltage > 3.5
                              ? Colors.orange
                              : Colors.red,
                    ),
                  ),
                ],
              ),
            ),
          if (isLeft)
            Chip(
              label: Text('Левый ${ble.leftDevice?.isConnected ?? false ? '✓' : '…'}'),
              backgroundColor: Colors.blue.shade800,
            )
          else if (isRight)
            Chip(
              label: Text('Правый ${ble.rightDevice?.isConnected ?? false ? '✓' : '…'}'),
              backgroundColor: Colors.purple.shade800,
            )
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
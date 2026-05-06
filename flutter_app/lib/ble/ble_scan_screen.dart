import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:provider/provider.dart';
import 'ble_manager.dart';

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

  const _DeviceTile({
    required this.result,
    required this.ble,
    required this.onAssign,
  });

  @override
  Widget build(BuildContext context) {
    final isLeft = ble.leftDevice?.device.id == result.device.id;
    final isRight = ble.rightDevice?.device.id == result.device.id;
    final name = result.device.platformName;

    return ListTile(
      leading: Icon(
        Icons.bluetooth,
        color: isLeft || isRight ? Colors.blue : Colors.grey,
      ),
      title: Text(name),
      subtitle: Text(result.device.id.id),
      trailing: isLeft
          ? Chip(
              label: Text('Левый ${ble.leftDevice?.isConnected ?? false ? '✓' : '…'}'),
              backgroundColor: Colors.blue.shade800,
            )
          : isRight
              ? Chip(
                  label: Text('Правый ${ble.rightDevice?.isConnected ?? false ? '✓' : '…'}'),
                  backgroundColor: Colors.purple.shade800,
                )
              : null,
      onTap: onAssign,
    );
  }
}
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:provider/provider.dart';
import 'ble_manager.dart';
import 'wt901_commander.dart';
import 'ui/status_bar.dart';
import 'ui/scanner_tile.dart';
import 'ui/connection_sheet.dart';
import 'ui/device_settings_sheet.dart';
import 'ui/rename_dialog.dart';

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
          StatusBar(leftDevice: ble.leftDevice, rightDevice: ble.rightDevice),
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
                    itemBuilder: (ctx, i) => ScannerTile(
                      result: devices[i],
                      leftDevice: ble.leftDevice,
                      rightDevice: ble.rightDevice,
                      voltage: ble.batteryLevels[devices[i].device.id.id],
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
      builder: (ctx) => ConnectionSheet(
        device: result.device,
        onLeft: () {
          ble.assignDevice('left', result.device);
        },
        onRight: () {
          ble.assignDevice('right', result.device);
        },
      ),
    );
  }

  void _showSensorSettings(ScanResult result, BleManager ble) {
    final commander = WT901Commander(result.device);
    showModalBottomSheet(
      context: context,
      builder: (ctx) => DeviceSettingsSheet(
        device: result.device,
        voltage: ble.batteryLevels[result.device.id.id],
        commander: commander,
        onRequestBattery: () async {
          await commander.requestBattery();
          if (ctx.mounted) Navigator.pop(ctx);
        },
        onRenamePressed: () => _showRenameDialog(result.device),
        onSetReturnRate: (code) async {
          await commander.setReturnRate(code);
          if (ctx.mounted) Navigator.pop(ctx);
        },
      ),
    );
  }

  void _showRenameDialog(BluetoothDevice device) {
    showDialog(
      context: context,
      builder: (ctx) => RenameDialog(device: device),
    );
  }
}
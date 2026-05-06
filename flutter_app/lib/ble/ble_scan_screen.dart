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
  bool _scanning = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _startScan();
  }

  @override
  void dispose() {
    // Don't use context in dispose — use BleManager directly
    super.dispose();
  }

  Future<void> _startScan() async {
    setState(() {
      _scanning = true;
      _error = null;
    });
    try {
      await context.read<BleManager>().startScan();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _scanning = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final ble = context.watch<BleManager>();

    return Scaffold(
      appBar: AppBar(title: const Text('Подключение IMU')),
      body: Column(
        children: [
          _StatusBar(ble: ble),
          if (_error != null)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              color: Colors.red.shade900,
              child: Text(_error!, style: const TextStyle(fontSize: 12)),
            ),
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
                    itemBuilder: (ctx, i) => _DeviceTile(
                      result: ble.scanResults[i],
                      ble: ble,
                      onAssign: () => _showAssignSheet(ble.scanResults[i], ble),
                    ),
                  ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _scanning ? null : _startScan,
                    icon: const Icon(Icons.search),
                    label: const Text('Сканировать'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: ble.leftDevice != null || ble.rightDevice != null
                        ? widget.onReady
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

  void _showAssignSheet(ScanResult result, BleManager ble) {
    final isLeft = ble.leftDevice?.device.id == result.device.id;
    final isRight = ble.rightDevice?.device.id == result.device.id;

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
}

class _StatusBar extends StatelessWidget {
  final BleManager ble;
  const _StatusBar({required this.ble});

  @override
  Widget build(BuildContext context) {
    final parts = <String>[];
    if (ble.leftDevice != null) parts.add('Левый ✓');
    if (ble.rightDevice != null) parts.add('Правый ✓');

    final color = ble.leftDevice != null && ble.rightDevice != null
        ? Colors.green.shade800
        : ble.leftDevice != null || ble.rightDevice != null
            ? Colors.orange.shade800
            : Colors.grey.shade800;

    final text = parts.isEmpty
        ? 'Выберите датчики (нажмите на устройство)'
        : '${parts.join('  ')}  —  можно продолжить';

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
      title: Text(name.isNotEmpty ? name : 'Неизвестное устройство'),
      subtitle: Text(result.device.id.id),
      trailing: isLeft
          ? Chip(label: const Text('Левый'), backgroundColor: Colors.blue.shade800)
          : isRight
              ? Chip(label: const Text('Правый'), backgroundColor: Colors.purple.shade800)
              : null,
      onTap: onAssign,
    );
  }
}
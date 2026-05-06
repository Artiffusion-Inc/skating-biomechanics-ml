import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../imu_device.dart';

class ScannerTile extends StatelessWidget {
  final ScanResult result;
  final IMUDevice? leftDevice;
  final IMUDevice? rightDevice;
  final double? voltage;
  final VoidCallback onAssign;
  final VoidCallback onSettings;

  const ScannerTile({
    super.key,
    required this.result,
    this.leftDevice,
    this.rightDevice,
    this.voltage,
    required this.onAssign,
    required this.onSettings,
  });

  @override
  Widget build(BuildContext context) {
    final isLeft = leftDevice?.device.id == result.device.id;
    final isRight = rightDevice?.device.id == result.device.id;
    final name = result.device.platformName;

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
              child: _BatteryIndicator(voltage: voltage!),
            ),
          if (isLeft)
            _SideChip(
              label: 'Левый ${leftDevice?.isConnected.value ?? false ? '✓' : '…'}',
              color: Colors.blue.shade800,
            )
          else if (isRight)
            _SideChip(
              label: 'Правый ${rightDevice?.isConnected.value ?? false ? '✓' : '…'}',
              color: Colors.purple.shade800,
            ),
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

class _BatteryIndicator extends StatelessWidget {
  final double voltage;
  const _BatteryIndicator({required this.voltage});

  Color get _color =>
      voltage > 3.7 ? Colors.green : voltage > 3.5 ? Colors.orange : Colors.red;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.battery_full, size: 14, color: _color),
        const SizedBox(width: 2),
        Text(
          '${voltage.toStringAsFixed(1)}V',
          style: TextStyle(fontSize: 11, color: _color),
        ),
      ],
    );
  }
}

class _SideChip extends StatelessWidget {
  final String label;
  final Color color;
  const _SideChip({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(label),
      backgroundColor: color,
    );
  }
}

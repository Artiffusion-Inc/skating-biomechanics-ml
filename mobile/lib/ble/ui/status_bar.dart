import 'package:flutter/material.dart';
import '../imu_device.dart';

class StatusBar extends StatelessWidget {
  final IMUDevice? leftDevice;
  final IMUDevice? rightDevice;

  const StatusBar({super.key, this.leftDevice, this.rightDevice});

  @override
  Widget build(BuildContext context) {
    final parts = <String>[];
    if (leftDevice != null) {
      parts.add('Левый ${leftDevice!.isConnected.value ? '✓' : '…'}');
    }
    if (rightDevice != null) {
      parts.add('Правый ${rightDevice!.isConnected.value ? '✓' : '…'}');
    }

    final leftOk = leftDevice?.isConnected.value ?? false;
    final rightOk = rightDevice?.isConnected.value ?? false;
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

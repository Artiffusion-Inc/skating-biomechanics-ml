import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

class ConnectionSheet extends StatelessWidget {
  final BluetoothDevice device;
  final VoidCallback onLeft;
  final VoidCallback onRight;

  const ConnectionSheet({
    super.key,
    required this.device,
    required this.onLeft,
    required this.onRight,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ListTile(
            title: Text(device.platformName),
            subtitle: Text(device.id.id),
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.skip_previous, color: Colors.blue),
            title: const Text('Левый датчик'),
            onTap: () {
              onLeft();
              Navigator.pop(context);
            },
          ),
          ListTile(
            leading: const Icon(Icons.skip_next, color: Colors.purple),
            title: const Text('Правый датчик'),
            onTap: () {
              onRight();
              Navigator.pop(context);
            },
          ),
        ],
      ),
    );
  }
}

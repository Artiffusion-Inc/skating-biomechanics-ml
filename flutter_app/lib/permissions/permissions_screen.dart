import 'package:flutter/material.dart';
import 'permission_service.dart';

class PermissionsScreen extends StatelessWidget {
  final VoidCallback onGranted;
  const PermissionsScreen({super.key, required this.onGranted});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('EdgeSense Capture')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.security, size: 64, color: Colors.white54),
              const SizedBox(height: 24),
              const Text('Для работы нужны разрешения', style: TextStyle(fontSize: 18)),
              const SizedBox(height: 8),
              const Text('Bluetooth, Камера, Микрофон, Местоположение',
                  style: TextStyle(color: Colors.white70)),
              const SizedBox(height: 32),
              FilledButton.icon(
                onPressed: () async {
                  final granted = await PermissionService().requestAll();
                  if (granted) onGranted();
                },
                icon: const Icon(Icons.verified_user),
                label: const Text('Предоставить разрешения'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
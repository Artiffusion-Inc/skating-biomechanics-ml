import 'package:flutter/material.dart';
import 'package:shadcn_flutter/shadcn_flutter.dart' as shad;
import 'permission_service.dart';
import '../../i18n/strings.g.dart';

class PermissionsScreen extends StatelessWidget {
  final VoidCallback onGranted;
  const PermissionsScreen({super.key, required this.onGranted});

  @override
  Widget build(BuildContext context) {
    final t = Translations.of(context);
    return Scaffold(
      appBar: AppBar(title: Text(t.app.title)),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.security, size: 64, color: Colors.white54),
              const SizedBox(height: 24),
              Text(
                t.permissions.required,
                style: const TextStyle(fontSize: 18),
              ),
              const SizedBox(height: 8),
              Text(
                t.permissions.list,
                style: const TextStyle(color: Colors.white70),
              ),
              const SizedBox(height: 32),
              shad.PrimaryButton(
                key: const Key('grantPermissionsBtn'),
                onPressed: () async {
                  final granted = await PermissionService().requestAll();
                  if (granted) onGranted();
                },
                leading: const Icon(Icons.verified_user),
                child: Text(t.permissions.grant),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

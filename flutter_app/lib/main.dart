import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'providers/app_providers.dart';
import 'theme/app_theme.dart';
import 'permissions/permissions_screen.dart';
import 'ble/ble_scan_screen.dart';
import 'camera/camera_ready_screen.dart';
import 'capture/capturing_screen.dart';
import 'capture/export_result_screen.dart';
import 'calibration/calibration_screen.dart';
import 'permissions/permission_service.dart';
import 'metrics/metrics_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AppProviders(child: EdgeSenseApp()));
}

class EdgeSenseApp extends StatelessWidget {
  const EdgeSenseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EdgeSense Capture',
      theme: AppTheme.dark,
      home: const AppShell(),
    );
  }
}

enum AppStep {
  permissions,
  bleScan,
  calibration,
  camera,
  capturing,
  exporting,
}

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  AppStep _step = AppStep.permissions;
  String? _exportPath;

  @override
  void initState() {
    super.initState();
    _checkPermissions();
  }

  Future<void> _checkPermissions() async {
    final granted = await PermissionService().requestAll();
    if (mounted) {
      setState(() => _step = granted ? AppStep.bleScan : AppStep.permissions);
    }
  }

  @override
  Widget build(BuildContext context) {
    return switch (_step) {
      AppStep.permissions => PermissionsScreen(
          onGranted: () => setState(() => _step = AppStep.bleScan),
        ),
      AppStep.bleScan => BleScanScreen(
          onReady: () => setState(() => _step = AppStep.calibration),
        ),
      AppStep.calibration => CalibrationScreen(
          onComplete: () => setState(() => _step = AppStep.camera),
        ),
      AppStep.camera => CameraReadyScreen(
          onStartCapture: () => setState(() => _step = AppStep.capturing),
        ),
      AppStep.capturing => CapturingScreen(
          onComplete: (path) {
            _exportPath = path;
            setState(() => _step = AppStep.exporting);
          },
        ),
      AppStep.exporting => ExportResultScreen(
          exportPath: _exportPath,
          onNewCapture: () => setState(() => _step = AppStep.bleScan),
        ),
    };
  }
}
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../ble/ble_manager.dart';
import '../camera/recorder.dart';
import '../capture/capture_provider.dart';
import '../calibration/calibration_service.dart';

class AppProviders extends StatelessWidget {
  final Widget child;
  const AppProviders({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => BleManager()),
        ChangeNotifierProvider(create: (_) => CameraRecorder()),
        ChangeNotifierProvider(
          create: (ctx) => CaptureProvider(
            bleManager: ctx.read<BleManager>(),
            cameraRecorder: ctx.read<CameraRecorder>(),
          ),
        ),
        ChangeNotifierProvider(create: (_) => CalibrationService()),
      ],
      child: child,
    );
  }
}

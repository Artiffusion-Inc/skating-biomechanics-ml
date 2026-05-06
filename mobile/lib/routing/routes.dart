import 'package:go_router/go_router.dart';

abstract final class AppRoutes {
  static const String permissions = '/';
  static const String bleScan = '/ble-scan';
  static const String calibration = '/calibration';
  static const String camera = '/camera';
  static const String capturing = '/capturing';
  static const String exporting = '/exporting';
}

extension GoRouterStateX on GoRouterState {
  String? get exportPath => uri.queryParameters['path'];
}

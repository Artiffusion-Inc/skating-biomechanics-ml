import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'routes.dart';
import '../permissions/permissions_screen.dart';
import '../permissions/permission_service.dart';
import '../ble/ble_scan_screen.dart';
import '../calibration/calibration_screen.dart';
import '../camera/camera_ready_screen.dart';
import '../capture/capturing_screen.dart';
import '../capture/export_result_screen.dart';

final _rootNavigatorKey = GlobalKey<NavigatorState>();

GoRouter buildRouter() {
  return GoRouter(
    navigatorKey: _rootNavigatorKey,
    initialLocation: AppRoutes.permissions,
    redirect: (context, state) async {
      final granted = await PermissionService.allGranted();
      final onPermissions = state.matchedLocation == AppRoutes.permissions;

      if (!granted && !onPermissions) {
        return AppRoutes.permissions;
      }
      if (granted && onPermissions) {
        return AppRoutes.bleScan;
      }
      return null;
    },
    routes: [
      GoRoute(
        path: AppRoutes.permissions,
        builder: (_, _) => PermissionsScreen(
          onGranted: () =>
              _rootNavigatorKey.currentContext?.go(AppRoutes.bleScan),
        ),
      ),
      GoRoute(
        path: AppRoutes.bleScan,
        builder: (_, _) => BleScanScreen(
          onReady: () =>
              _rootNavigatorKey.currentContext?.go(AppRoutes.calibration),
        ),
      ),
      GoRoute(
        path: AppRoutes.calibration,
        builder: (_, _) => CalibrationScreen(
          onComplete: () =>
              _rootNavigatorKey.currentContext?.go(AppRoutes.camera),
        ),
      ),
      GoRoute(
        path: AppRoutes.camera,
        builder: (_, _) => CameraReadyScreen(
          onStartCapture: () =>
              _rootNavigatorKey.currentContext?.go(AppRoutes.capturing),
        ),
      ),
      GoRoute(
        path: AppRoutes.capturing,
        builder: (_, _) => CapturingScreen(
          onComplete: (path) => _rootNavigatorKey.currentContext?.go(
            '${AppRoutes.exporting}?path=${Uri.encodeComponent(path ?? '')}',
          ),
        ),
      ),
      GoRoute(
        path: AppRoutes.exporting,
        builder: (context, state) {
          final path = state.uri.queryParameters['path'];
          return ExportResultScreen(
            exportPath: path,
            onNewCapture: () => context.go(AppRoutes.bleScan),
          );
        },
      ),
    ],
  );
}

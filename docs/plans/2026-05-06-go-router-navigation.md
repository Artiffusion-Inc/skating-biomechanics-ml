# GoRouter Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace linear `AppShell` enum with declarative `GoRouter` routing, enabling back button support, deep linking, and proper navigation stack.

**Architecture:** Define routes per `AppStep` (permissions, bleScan, calibration, camera, capturing, exporting). Each route maps to existing screen widget. `AppShell` replaced by `GoRouter` with `MaterialApp.router`. Deep link handler redirects `/` → permissions if needed.

**Tech Stack:** `go_router`, `provider`, existing screen widgets

---

## File Structure

| File | Responsibility |
|------|----------------|
| `mobile/lib/routing/app_router.dart` | Route definitions, guards, redirect logic |
| `mobile/lib/routing/routes.dart` | Route path constants and typed extensions |
| `mobile/lib/main.dart` | Bootstraps `MaterialApp.router` instead of `AppShell` |
| `mobile/lib/providers/app_providers.dart` | Unchanged — still provides BleManager, CameraRecorder, etc. |
| `mobile/lib/screens/` (existing) | All screens kept in place, only navigation callbacks change |
| `mobile/test/routing/app_router_test.dart` | Unit tests: redirect logic, route matching |

---

## Task 1: Add `go_router` Dependency

**Files:**
- Modify: `mobile/pubspec.yaml`

- [ ] **Step 1: Add dependency**

```yaml
dependencies:
  flutter:
    sdk: flutter
  go_router: ^14.1.0
  # ... rest unchanged
```

- [ ] **Step 2: Get packages**

Run: `cd mobile && flutter pub get`
Expected: `go_router` resolved.

- [ ] **Step 3: Commit**

```bash
git add mobile/pubspec.yaml
git commit -m "chore(deps): add go_router for declarative navigation"
```

---

## Task 2: Route Constants and Typed Helpers

**Files:**
- Create: `mobile/lib/routing/routes.dart`

- [ ] **Step 1: Write route constants**

```dart
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
```

- [ ] **Step 2: Commit**

```bash
git add mobile/lib/routing/routes.dart
git commit -m "feat(routing): add route constants and typed helpers"
```

---

## Task 3: Build AppRouter with Redirect Guard

**Files:**
- Create: `mobile/lib/routing/app_router.dart`

- [ ] **Step 1: Write router configuration**

```dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';

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
      final granted = await PermissionService().checkAll();
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
        builder: (_, __) => PermissionsScreen(
          onGranted: () => _rootNavigatorKey.currentContext?.go(AppRoutes.bleScan),
        ),
      ),
      GoRoute(
        path: AppRoutes.bleScan,
        builder: (_, __) => BleScanScreen(
          onReady: () => _rootNavigatorKey.currentContext?.go(AppRoutes.calibration),
        ),
      ),
      GoRoute(
        path: AppRoutes.calibration,
        builder: (_, __) => CalibrationScreen(
          onComplete: () => _rootNavigatorKey.currentContext?.go(AppRoutes.camera),
        ),
      ),
      GoRoute(
        path: AppRoutes.camera,
        builder: (_, __) => CameraReadyScreen(
          onStartCapture: () => _rootNavigatorKey.currentContext?.go(AppRoutes.capturing),
        ),
      ),
      GoRoute(
        path: AppRoutes.capturing,
        builder: (_, __) => CapturingScreen(
          onComplete: (path) => _rootNavigatorKey.currentContext?.go(
            '${AppRoutes.exporting}?path=$path',
          ),
        ),
      ),
      GoRoute(
        path: AppRoutes.exporting,
        builder: (context, state) {
          final path = state.exportPath;
          return ExportResultScreen(
            exportPath: path,
            onNewCapture: () => context.go(AppRoutes.bleScan),
          );
        },
      ),
    ],
  );
}
```

- [ ] **Step 2: Add `checkAll` to PermissionService**

In `mobile/lib/permissions/permission_service.dart`, add:

```dart
Future<bool> checkAll() async {
  return await Permission.bluetoothScan.isGranted &&
      await Permission.bluetoothConnect.isGranted &&
      await Permission.camera.isGranted &&
      await Permission.location.isGranted;
}
```

- [ ] **Step 3: Commit**

```bash
git add mobile/lib/routing/app_router.dart mobile/lib/permissions/permission_service.dart
git commit -m "feat(routing): add GoRouter with permission guard redirect"
```

---

## Task 4: Replace AppShell with MaterialApp.router

**Files:**
- Modify: `mobile/lib/main.dart`

- [ ] **Step 1: Replace main.dart contents**

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'providers/app_providers.dart';
import 'theme/app_theme.dart';
import 'routing/app_router.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AppProviders(child: EdgeSenseApp()));
}

class EdgeSenseApp extends StatelessWidget {
  const EdgeSenseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'EdgeSense Capture',
      theme: AppTheme.dark,
      routerConfig: buildRouter(),
    );
  }
}
```

- [ ] **Step 2: Delete AppShell**

`AppShell` class (old `main.dart` lines 33-93) is now dead code. Remove it.

- [ ] **Step 3: Write router test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:edgesense_capture/routing/app_router.dart';
import 'package:edgesense_capture/routing/routes.dart';

void main() {
  group('AppRouter', () {
    test('initial location is permissions', () {
      final router = buildRouter();
      expect(router.routeInformationProvider.value.location, AppRoutes.permissions);
    });

    test('route paths match expected', () {
      final router = buildRouter();
      final paths = router.configuration.routes.map((r) => (r as GoRoute).path).toList();
      expect(paths, contains(AppRoutes.permissions));
      expect(paths, contains(AppRoutes.bleScan));
      expect(paths, contains(AppRoutes.calibration));
      expect(paths, contains(AppRoutes.camera));
      expect(paths, contains(AppRoutes.capturing));
      expect(paths, contains(AppRoutes.exporting));
    });
  });
}
```

- [ ] **Step 4: Run tests**

Run: `cd mobile && flutter test test/routing/app_router_test.dart -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/lib/main.dart mobile/test/routing/app_router_test.dart
git rm mobile/lib/main.dart  # if AppShell was separate file, remove it
git commit -m "feat(routing): replace AppShell enum with GoRouter"
```

---

## Self-Review

1. **Spec coverage:**
   - GoRouter replaces `AppShell` enum → Task 4 ✅
   - Native back button via GoRouter stack → implicit ✅
   - Deep linking via route paths → Task 3 ✅

2. **Placeholder scan:** No TBD/TODO. All code shown. ✅

3. **Type consistency:** `AppRoutes` used in `routes.dart` and `app_router.dart`. `PermissionService.checkAll` matches `redirect` signature. ✅

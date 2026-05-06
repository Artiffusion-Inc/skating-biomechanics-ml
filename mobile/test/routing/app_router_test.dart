import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:edgesense_capture/routing/app_router.dart';
import 'package:edgesense_capture/routing/routes.dart';

void main() {
  group('AppRouter', () {
    test('initial location is permissions', () {
      final router = buildRouter();
      expect(router.routeInformationProvider.value.uri.path, AppRoutes.permissions);
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

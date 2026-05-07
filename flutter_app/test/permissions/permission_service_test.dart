import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/permissions/permission_service.dart';

void main() {
  group('PermissionService', () {
    test('can be instantiated', () {
      final service = PermissionService();
      expect(service, isA<PermissionService>());
    });
  });
}

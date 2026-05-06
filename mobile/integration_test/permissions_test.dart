import 'package:flutter_test/flutter_test.dart';
import 'package:patrol/patrol.dart';
import 'package:edgesense_capture/main.dart' as app;

void main() {
  patrolTest(
    'grants permissions and navigates to BLE scan',
    (PatrolTester $) async {
      await app.main();
      await $.pumpAndSettle();

      // Permissions screen
      await $('Предоставить разрешения').tap();

      // Patrol auto-grants Android permissions via orchestrator
      // Wait for BLE scan screen
      await $('Подключение IMU').waitUntilExists();

      // Tap scan
      await $('Сканировать').tap();
      await $.pumpAndSettle(duration: const Duration(seconds: 6));

      // Verify list or empty state appears
      expect($('Сканировать снова').exists || $('ListView').exists, true);
    },
  );
}

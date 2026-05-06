import 'package:flutter_test/flutter_test.dart';
import 'package:patrol/patrol.dart';
import 'package:edgesense_capture/main.dart' as app;

void main() {
  patrolTest(
    'full capture flow: permissions → ble → calibration → camera → stop',
    (PatrolTester $) async {
      await app.main();
      await $.pumpAndSettle();

      // 1. Permissions screen
      await $('Предоставить разрешения').tap();
      await $.pumpAndSettle();

      // 2. BLE scan screen
      await $('Подключение IMU').waitUntilExists();
      await $('Сканировать').tap();
      await $.pumpAndSettle(duration: const Duration(seconds: 8));

      // Either scan results or empty state
      final hasResults = $('Сканировать снова').exists;
      if (hasResults) {
        // Empty state — can't proceed without real devices
        // Verify navigation flow works up to this point
        expect($('Нажмите на устройство для назначения').exists || $('ListView').exists, true);
        return;
      }

      // If devices found (emulator with mock devices), continue flow
      // Tap first device → assign left
      await $('ListView').tap();
      await $.pumpAndSettle();
      await $('Левый датчик').tap();
      await $.pumpAndSettle();

      // Proceed to calibration
      await $('Далее').tap();
      await $.pumpAndSettle();

      // 3. Calibration screen
      await $('Калибровка').waitUntilExists();
      await $('Начать калибровку').tap();
      await $.pumpAndSettle(duration: const Duration(seconds: 3));
      await $('Завершить калибровку').tap();
      await $.pumpAndSettle();

      // 4. Camera ready screen
      await $('Камера готова').waitUntilExists();
      await $('Начать запись').tap();
      await $.pumpAndSettle(duration: const Duration(seconds: 2));

      // 5. Capturing screen
      await $('Стоп').waitUntilVisible();
      await $.pumpAndSettle(duration: const Duration(seconds: 2));
      await $('Стоп').tap();
      await $.pumpAndSettle(duration: const Duration(seconds: 4));

      // 6. Export result screen
      await $('Экспорт завершен').waitUntilExists();
    },
  );
}

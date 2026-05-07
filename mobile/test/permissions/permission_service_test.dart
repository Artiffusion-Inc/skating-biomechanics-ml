import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:edgesense_capture/permissions/permission_service.dart';
import 'package:edgesense_capture/permissions/permission_handler.dart';

class MockPermissionGateway extends Mock implements PermissionGateway {}

void main() {
  group('PermissionService', () {
    late MockPermissionGateway gateway;
    late PermissionService service;

    setUp(() {
      gateway = MockPermissionGateway();
      service = PermissionService(gateway: gateway);

      when(
        () => gateway.requestLocation(),
      ).thenAnswer((_) async => PermissionStatus.granted);
      when(
        () => gateway.requestBluetoothScan(),
      ).thenAnswer((_) async => PermissionStatus.granted);
      when(
        () => gateway.requestBluetoothConnect(),
      ).thenAnswer((_) async => PermissionStatus.granted);
      when(
        () => gateway.requestCamera(),
      ).thenAnswer((_) async => PermissionStatus.granted);
      when(
        () => gateway.requestMicrophone(),
      ).thenAnswer((_) async => PermissionStatus.granted);
    });

    test('requestAll returns true when all granted', () async {
      final result = await service.requestAll();
      expect(result, true);
    });

    test('requestAll returns false when location denied', () async {
      when(
        () => gateway.requestLocation(),
      ).thenAnswer((_) async => PermissionStatus.denied);
      final result = await service.requestAll();
      expect(result, false);
    });

    test('requestAll returns false when bluetooth scan denied', () async {
      when(
        () => gateway.requestBluetoothScan(),
      ).thenAnswer((_) async => PermissionStatus.denied);
      final result = await service.requestAll();
      expect(result, false);
    });

    test('requestAll returns false when camera denied', () async {
      when(
        () => gateway.requestCamera(),
      ).thenAnswer((_) async => PermissionStatus.denied);
      final result = await service.requestAll();
      expect(result, false);
    });

    test('requestAll returns false when microphone denied', () async {
      when(
        () => gateway.requestMicrophone(),
      ).thenAnswer((_) async => PermissionStatus.denied);
      final result = await service.requestAll();
      expect(result, false);
    });

    test('requestAll calls all five permissions', () async {
      await service.requestAll();

      verify(() => gateway.requestLocation()).called(1);
      verify(() => gateway.requestBluetoothScan()).called(1);
      verify(() => gateway.requestBluetoothConnect()).called(1);
      verify(() => gateway.requestCamera()).called(1);
      verify(() => gateway.requestMicrophone()).called(1);
    });
  });
}

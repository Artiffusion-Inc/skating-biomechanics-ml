import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../../../i18n/strings.g.dart';
import '../wt901_commander.dart';

class DeviceSettingsSheet extends StatelessWidget {
  final BluetoothDevice device;
  final double? voltage;
  final WT901Commander commander;
  final VoidCallback onRequestBattery;
  final VoidCallback onRenamePressed;
  final void Function(int code) onSetReturnRate;

  const DeviceSettingsSheet({
    super.key,
    required this.device,
    this.voltage,
    required this.commander,
    required this.onRequestBattery,
    required this.onRenamePressed,
    required this.onSetReturnRate,
  });

  @override
  Widget build(BuildContext context) {
    final t = Translations.of(context);
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              device.platformName,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            Text(device.remoteId.str, style: const TextStyle(fontSize: 12, color: Colors.white70)),
            const SizedBox(height: 16),
            _BatteryTile(
              voltage: voltage,
              onRequest: onRequestBattery,
            ),
            const Divider(),
            _ReturnRateTile(onSelected: onSetReturnRate),
            ListTile(
              leading: const Icon(Icons.edit),
              title: Text(t.ble.rename.title),
              trailing: FilledButton.tonal(
                onPressed: onRenamePressed,
                child: Text(t.ble.rename.action),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _BatteryTile extends StatelessWidget {
  final double? voltage;
  final VoidCallback onRequest;
  const _BatteryTile({this.voltage, required this.onRequest});

  Color get _color =>
      voltage == null ? Colors.grey : voltage! > 3.7 ? Colors.green : voltage! > 3.5 ? Colors.orange : Colors.red;

  @override
  Widget build(BuildContext context) {
    final t = Translations.of(context);
    return ListTile(
      leading: Icon(Icons.battery_full, color: _color),
      title: Text(t.ble.battery.title),
      subtitle: Text(voltage == null ? t.ble.battery.unknown : '${voltage!.toStringAsFixed(2)} В'),
      trailing: TextButton(
        onPressed: onRequest,
        child: Text(t.ble.battery.request),
      ),
    );
  }
}

class _ReturnRateTile extends StatelessWidget {
  final void Function(int code) onSelected;
  const _ReturnRateTile({required this.onSelected});

  @override
  Widget build(BuildContext context) {
    final t = Translations.of(context);
    return ListTile(
      leading: const Icon(Icons.speed),
      title: Text(t.ble.returnRate.title),
      subtitle: Text(t.ble.returnRate.hint),
      trailing: DropdownButton<int>(
        value: null,
        hint: Text(t.ble.returnRate.select),
        underline: const SizedBox.shrink(),
        items: const [
          DropdownMenuItem(value: 0x01, child: Text('0.2 Гц')),
          DropdownMenuItem(value: 0x02, child: Text('0.5 Гц')),
          DropdownMenuItem(value: 0x03, child: Text('1 Гц')),
          DropdownMenuItem(value: 0x04, child: Text('2 Гц')),
          DropdownMenuItem(value: 0x05, child: Text('5 Гц')),
          DropdownMenuItem(value: 0x06, child: Text('10 Гц')),
          DropdownMenuItem(value: 0x07, child: Text('20 Гц')),
          DropdownMenuItem(value: 0x08, child: Text('50 Гц')),
          DropdownMenuItem(value: 0x09, child: Text('100 Гц')),
          DropdownMenuItem(value: 0x0B, child: Text('200 Гц')),
        ],
        onChanged: (code) {
          if (code != null) onSelected(code);
        },
      ),
    );
  }
}

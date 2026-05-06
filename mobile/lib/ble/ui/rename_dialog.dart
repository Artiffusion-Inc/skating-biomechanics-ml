import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../wt901_commander.dart';

class RenameDialog extends StatefulWidget {
  final BluetoothDevice device;
  const RenameDialog({super.key, required this.device});

  @override
  State<RenameDialog> createState() => _RenameDialogState();
}

class _RenameDialogState extends State<RenameDialog> {
  final _ctrl = TextEditingController();

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final id = int.tryParse(_ctrl.text);
    if (id == null || id < 0 || id > 255) return;
    final commander = WT901Commander(widget.device);
    await commander.rename(id);
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Переименовать датчик'),
      content: TextField(
        controller: _ctrl,
        decoration: const InputDecoration(
          labelText: 'Новый ID (0-255)',
          hintText: 'Например: 1',
        ),
        keyboardType: TextInputType.number,
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Отмена'),
        ),
        FilledButton(
          onPressed: _save,
          child: const Text('Сохранить'),
        ),
      ],
    );
  }
}

import 'package:flutter/material.dart';

class ExportResultScreen extends StatelessWidget {
  final String? exportPath;
  final VoidCallback onNewCapture;
  const ExportResultScreen({
    super.key,
    required this.exportPath,
    required this.onNewCapture,
  });

  @override
  Widget build(BuildContext context) {
    final success = exportPath != null;

    return Scaffold(
      appBar: AppBar(title: const Text('Результат')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                success ? Icons.check_circle : Icons.error,
                size: 72,
                color: success ? Colors.green : Colors.red,
              ),
              const SizedBox(height: 24),
              Text(
                success ? 'Запись сохранена' : 'Ошибка при сохранении',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              if (success) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.white10,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: SelectableText(
                    exportPath!,
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                  ),
                ),
              ],
              const SizedBox(height: 32),
              FilledButton.icon(
                onPressed: onNewCapture,
                icon: const Icon(Icons.add),
                label: const Text('Новая запись'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
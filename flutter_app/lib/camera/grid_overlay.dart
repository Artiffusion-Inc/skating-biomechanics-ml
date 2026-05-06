import 'package:flutter/material.dart';

class GridOverlay extends StatelessWidget {
  const GridOverlay({super.key});

  @override
  Widget build(BuildContext context) {
    return const CustomPaint(painter: _GridPainter(), size: Size.infinite);
  }
}

class _GridPainter extends CustomPainter {
  const _GridPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.3)
      ..strokeWidth = 1;
    final dx = size.width / 3;
    final dy = size.height / 3;
    canvas.drawLine(Offset(dx, 0), Offset(dx, size.height), paint);
    canvas.drawLine(Offset(2 * dx, 0), Offset(2 * dx, size.height), paint);
    canvas.drawLine(Offset(0, dy), Offset(size.width, dy), paint);
    canvas.drawLine(Offset(0, 2 * dy), Offset(size.width, 2 * dy), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

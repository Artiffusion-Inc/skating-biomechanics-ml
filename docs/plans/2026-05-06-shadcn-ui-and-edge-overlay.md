# Shadcn UI and EdgeOverlay "No Data" Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Shadcn Flutter design tokens for consistent UI. Add "No Data" visual alert to `EdgeOverlay` when quaternion hasn't arrived > 100ms. Ensure no Skia-specific shaders (Impeller is default).

**Architecture:** `ShadThemeData` with custom color scheme matching existing `AppTheme.dark`. `EdgeOverlay` gets `Timer`-based staleness detection: if `DateTime.now().difference(lastPacketTime) > 100ms`, show red border + warning icon. All drawing uses standard Flutter primitives (no custom shaders).

**Tech Stack:** `shadcn_ui`, existing `EdgeOverlay`, `provider`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `mobile/lib/theme/shad_theme.dart` | `ShadThemeData` configuration |
| `mobile/lib/theme/app_theme.dart` | Update to use `ShadTheme` or merge |
| `mobile/lib/overlay/edge_overlay.dart` | Add staleness detection + visual alert |
| `mobile/lib/capture/capturing_screen.dart` | Pass `lastPacketTime` to `EdgeOverlay` |
| `mobile/test/overlay/edge_overlay_test.dart` | Widget test: alert appears after 100ms |

---

## Task 1: Add `shadcn_ui` Dependency

**Files:**
- Modify: `mobile/pubspec.yaml`

- [ ] **Step 1: Add dependency**

```yaml
dependencies:
  flutter:
    sdk: flutter
  shadcn_ui: ^0.9.0
  # ... keep existing
```

- [ ] **Step 2: Get packages**

Run: `cd mobile && flutter pub get`
Expected: `shadcn_ui` resolved.

- [ ] **Step 3: Commit**

```bash
git add mobile/pubspec.yaml
git commit -m "chore(deps): add shadcn_ui for consistent design tokens"
```

---

## Task 2: Configure ShadTheme

**Files:**
- Create: `mobile/lib/theme/shad_theme.dart`
- Modify: `mobile/lib/theme/app_theme.dart`

- [ ] **Step 1: Write ShadTheme configuration**

```dart
import 'package:flutter/material.dart';
import 'package:shadcn_ui/shadcn_ui.dart';

class AppShadTheme {
  static ShadThemeData get dark {
    return ShadThemeData(
      brightness: Brightness.dark,
      colorScheme: const ShadColorScheme.dark(
        background: Color(0xFF121212),
        foreground: Color(0xFFE0E0E0),
        primary: Color(0xFF1E88E5),
        primaryForeground: Color(0xFFFFFFFF),
        destructive: Color(0xFFEF4444),
        destructiveForeground: Color(0xFFFFFFFF),
      ),
      radius: BorderRadius.circular(8),
    );
  }
}
```

- [ ] **Step 2: Update app_theme.dart to export**

```dart
export 'shad_theme.dart' show AppShadTheme;
```

- [ ] **Step 3: Commit**

```bash
git add mobile/lib/theme/shad_theme.dart mobile/lib/theme/app_theme.dart
git commit -m "feat(theme): add ShadThemeData with dark color scheme"
```

---

## Task 3: EdgeOverlay Staleness Detection

**Files:**
- Modify: `mobile/lib/overlay/edge_overlay.dart`

- [ ] **Step 1: Add Timer-based staleness check**

```dart
import 'dart:async';
import 'package:flutter/material.dart';

class EdgeOverlay extends StatefulWidget {
  final double leftAngle;
  final double rightAngle;
  final DateTime? lastPacketTime;

  const EdgeOverlay({
    super.key,
    required this.leftAngle,
    required this.rightAngle,
    this.lastPacketTime,
  });

  @override
  State<EdgeOverlay> createState() => _EdgeOverlayState();
}

class _EdgeOverlayState extends State<EdgeOverlay> {
  bool _stale = false;
  Timer? _timer;

  @override
  void didUpdateWidget(covariant EdgeOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    _updateStale();
  }

  @override
  void initState() {
    super.initState();
    _updateStale();
  }

  void _updateStale() {
    _timer?.cancel();
    final last = widget.lastPacketTime;
    if (last == null) {
      setState(() => _stale = true);
      return;
    }
    final elapsed = DateTime.now().difference(last);
    if (elapsed > const Duration(milliseconds: 100)) {
      setState(() => _stale = true);
    } else {
      setState(() => _stale = false);
      _timer = Timer(
        const Duration(milliseconds: 100) - elapsed,
        () => setState(() => _stale = true),
      );
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 40,
      left: 20,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(8),
          border: _stale
              ? Border.all(color: Colors.red, width: 2)
              : null,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text('L: ${widget.leftAngle.toStringAsFixed(1)}°',
                    style: const TextStyle(color: Colors.white, fontSize: 18)),
                if (_stale) ...[
                  const SizedBox(width: 8),
                  const Icon(Icons.warning, color: Colors.red, size: 16),
                ],
              ],
            ),
            Text('R: ${widget.rightAngle.toStringAsFixed(1)}°',
                style: const TextStyle(color: Colors.white, fontSize: 18)),
            if (_stale)
              const Text(
                'Нет данных',
                style: TextStyle(color: Colors.red, fontSize: 12),
              ),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 2: Update CapturingScreen to pass lastPacketTime**

In `mobile/lib/capture/capturing_screen.dart`, track `lastPacketTime` from `CaptureProvider` or repository callback:

```dart
// Wherever EdgeOverlay is instantiated:
EdgeOverlay(
  leftAngle: leftAngle,
  rightAngle: rightAngle,
  lastPacketTime: provider.lastPacketTime, // add to provider if needed
)
```

- [ ] **Step 3: Commit**

```bash
git add mobile/lib/overlay/edge_overlay.dart mobile/lib/capture/capturing_screen.dart
git commit -m "feat(overlay): add staleness detection and No Data visual alert"
```

---

## Task 4: Verify No Skia Shaders

**Files:**
- Check: all `CustomPainter` or `Shader` usage in `mobile/lib/`

- [ ] **Step 1: Search for custom shaders**

Run: `grep -r "Shader\|FragmentShader\|Paint.shader" mobile/lib/ || echo "None found"`
Expected: "None found" or only standard gradient shaders (safe).

- [ ] **Step 2: Commit**

```bash
git commit -m "chore(ui): verify no Skia-specific shaders for Impeller compatibility"
```

---

## Task 5: Widget Test for Staleness

**Files:**
- Create: `mobile/test/overlay/edge_overlay_test.dart`

- [ ] **Step 1: Write test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:edgesense_capture/overlay/edge_overlay.dart';

void main() {
  testWidgets('shows alert when lastPacketTime is stale', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: EdgeOverlay(
            leftAngle: 10,
            rightAngle: 20,
            lastPacketTime: null,
          ),
        ),
      ),
    );

    expect(find.text('Нет данных'), findsOneWidget);
    expect(find.byIcon(Icons.warning), findsOneWidget);
  });

  testWidgets('shows alert after 100ms', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: EdgeOverlay(
            leftAngle: 10,
            rightAngle: 20,
            lastPacketTime: DateTime.now().subtract(const Duration(milliseconds: 150)),
          ),
        ),
      ),
    );

    expect(find.text('Нет данных'), findsOneWidget);
  });

  testWidgets('no alert when fresh', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: EdgeOverlay(
            leftAngle: 10,
            rightAngle: 20,
            lastPacketTime: DateTime.now(),
          ),
        ),
      ),
    );

    expect(find.text('Нет данных'), findsNothing);
  });
}
```

- [ ] **Step 2: Run tests**

Run: `cd mobile && flutter test test/overlay/edge_overlay_test.dart -v`
Expected: 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add mobile/test/overlay/edge_overlay_test.dart
git commit -m "test(overlay): verify EdgeOverlay staleness alert behavior"
```

---

## Self-Review

1. **Spec coverage:**
   - Shadcn Flutter theme applied → Task 2 ✅
   - EdgeOverlay "No Data" indicator after 100ms → Task 3 ✅
   - Impeller-compatible (no Skia shaders) → Task 4 ✅

2. **Placeholder scan:** No TBD/TODO. ✅

3. **Type consistency:** `EdgeOverlay` uses same `leftAngle`/`rightAngle` props, adds optional `lastPacketTime`. ✅

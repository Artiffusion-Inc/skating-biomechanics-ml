# Developer Utilities Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FlutterGen, Very Good CLI, and Melos to eliminate boilerplate, enforce feature structure, and prepare for multi-package growth.

**Architecture:** FlutterGen generates asset constants from `pubspec.yaml`. Very Good CLI creates features with consistent folder layout. Melos manages workspace if BLE parser is extracted later. All tools run via `go-task` (Taskfile.yaml).

**Tech Stack:** `flutter_gen`, `very_good_cli`, `melos`, `go-task`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `mobile/pubspec.yaml` | Add `flutter_gen_runner` dev dependency |
| `mobile/flutter_gen.yaml` | FlutterGen config: output path, style |
| `mobile/Taskfile.yaml` | Tasks: `gen`, `feature-create`, `bootstrap`, `analyze` |
| `mobile/.verygoodrc` | Very Good CLI config: default org, output dir |
| `mobile/melos.yaml` | Melos workspace config (prepared, active when packages extracted) |
| `mobile/lib/gen/assets.gen.dart` | Generated asset references |
| `mobile/test/gen/assets_test.dart` | Verify generated paths match actual files |

---

## Task 1: Add FlutterGen

**Files:**
- Modify: `mobile/pubspec.yaml`
- Create: `mobile/flutter_gen.yaml`

- [ ] **Step 1: Add dev dependency**

```yaml
dev_dependencies:
  flutter_gen_runner: ^5.5.0
  build_runner: ^2.4.0
  # ... keep existing
```

- [ ] **Step 2: Add asset to pubspec**

```yaml
flutter:
  uses-material-design: true
  assets:
    - assets/images/
```

- [ ] **Step 3: Write flutter_gen.yaml**

```yaml
flutter_gen:
  output: lib/gen/
  line_length: 80
  integrations:
    flutter_svg: true
  assets:
    outputs:
      class_name: Assets
      package_parameter_enabled: false
      style: dot-delimiter
```

- [ ] **Step 4: Run generator**

Run: `cd mobile && flutter pub get && dart run build_runner build --delete-conflicting-outputs`
Expected: `lib/gen/assets.gen.dart` created.

- [ ] **Step 5: Verify generated import works**

```dart
import 'gen/assets.gen.dart';
// Assets.images.logo.path instead of 'assets/images/logo.png'
```

- [ ] **Step 6: Commit**

```bash
git add mobile/pubspec.yaml mobile/flutter_gen.yaml mobile/lib/gen/assets.gen.dart
git commit -m "chore(tools): add FlutterGen for type-safe asset references"
```

---

## Task 2: Add Very Good CLI

**Files:**
- Modify: `mobile/Taskfile.yaml`
- Create: `mobile/.verygoodrc`

- [ ] **Step 1: Install CLI globally**

Run: `dart pub global activate very_good_cli`
Expected: `very_good` command available.

- [ ] **Step 2: Write .verygoodrc**

```yaml
org_name: com.artiffusion
project_name: edgesense_capture
output_directory: lib/features
```

- [ ] **Step 3: Add go-task command**

In `mobile/Taskfile.yaml` (create if absent):

```yaml
version: '3'

tasks:
  gen:
    desc: Generate code (FlutterGen + slang)
    cmds:
      - dart run build_runner build --delete-conflicting-outputs
      - dart run slang

  feature:
    desc: Create new feature via Very Good CLI
    cmds:
      - very_good create flutter_package {{.CLI_ARGS}} --output-directory lib/features

  bootstrap:
    desc: Install dependencies and generate code
    cmds:
      - flutter pub get
      - task gen

  analyze:
    desc: Analyze and format check
    cmds:
      - flutter analyze
      - dart format --output=none --set-exit-if-changed lib/ test/

  test:
    desc: Run all tests
    cmds:
      - flutter test
```

- [ ] **Step 4: Commit**

```bash
git add mobile/.verygoodrc mobile/Taskfile.yaml
git commit -m "chore(tools): add Very Good CLI config and go-task commands"
```

---

## Task 3: Prepare Melos Workspace

**Files:**
- Create: `mobile/melos.yaml`

- [ ] **Step 1: Write melos.yaml**

```yaml
name: edgesense_capture
packages:
  - .
  - packages/*

scripts:
  analyze:
    exec: flutter analyze
  test:
    exec: flutter test
  gen:
    exec: dart run build_runner build --delete-conflicting-outputs
```

- [ ] **Step 2: Commit (preparatory)**

```bash
git add mobile/melos.yaml
git commit -m "chore(tools): add Melos workspace config for future package extraction"
```

---

## Task 4: Replace Manual Asset Strings with FlutterGen

**Files:**
- Modify: any file referencing asset paths (if assets exist)

- [ ] **Step 1: Replace hardcoded paths**

Example:

```dart
// Before:
Image.asset('assets/images/logo.png')

// After:
import 'gen/assets.gen.dart';
Image.asset(Assets.images.logo.path)
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor(assets): replace string paths with FlutterGen constants"
```

---

## Self-Review

1. **Spec coverage:**
   - FlutterGen generates asset refs → Task 1 ✅
   - Very Good CLI for feature creation → Task 2 ✅
   - Melos prepared for multi-package → Task 3 ✅

2. **Placeholder scan:** No TBD/TODO. ✅

3. **Type consistency:** `Assets` class generated by FlutterGen, used in widget code. ✅

import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'providers/app_providers.dart';
import 'theme/app_theme.dart';
import 'routing/app_router.dart';
import 'i18n/strings.g.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  LocaleSettings.useDeviceLocale();
  runApp(TranslationProvider(child: const AppProviders(child: EdgeSenseApp())));
}

class EdgeSenseApp extends StatelessWidget {
  const EdgeSenseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'EdgeSense Capture',
      theme: AppTheme.dark,
      routerConfig: buildRouter(),
      locale: TranslationProvider.of(context).flutterLocale,
      supportedLocales: AppLocaleUtils.supportedLocales,
      localizationsDelegates: GlobalMaterialLocalizations.delegates,
    );
  }
}

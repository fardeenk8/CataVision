import 'package:flutter/material.dart';

import 'app_theme_scope.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const CataractDetectionApp());
}

class CataractDetectionApp extends StatefulWidget {
  const CataractDetectionApp({super.key});

  @override
  State<CataractDetectionApp> createState() => _CataractDetectionAppState();
}

class _CataractDetectionAppState extends State<CataractDetectionApp> with WidgetsBindingObserver {
  ThemeMode _themeMode = ThemeMode.system;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangePlatformBrightness() {
    super.didChangePlatformBrightness();
    setState(() {});
  }

  void _toggleTheme() {
    setState(() {
      final effective = effectiveThemeBrightness(_themeMode);
      _themeMode = effective == Brightness.dark ? ThemeMode.light : ThemeMode.dark;
    });
  }

  @override
  Widget build(BuildContext context) {
    final platformBrightness = WidgetsBinding.instance.platformDispatcher.platformBrightness;
    return AppThemeScope(
      themeMode: _themeMode,
      platformBrightness: platformBrightness,
      toggleTheme: _toggleTheme,
      child: MaterialApp(
        title: 'Cataract Detection',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0EA5E9), brightness: Brightness.light),
          useMaterial3: true,
          fontFamily: 'Roboto',
        ),
        darkTheme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0EA5E9), brightness: Brightness.dark),
          useMaterial3: true,
          fontFamily: 'Roboto',
        ),
        themeMode: _themeMode,
        home: const HomeScreen(),
      ),
    );
  }
}

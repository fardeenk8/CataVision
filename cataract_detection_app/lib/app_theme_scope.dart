import 'package:flutter/material.dart';

Brightness effectiveThemeBrightness(ThemeMode mode) {
  switch (mode) {
    case ThemeMode.dark:
      return Brightness.dark;
    case ThemeMode.light:
      return Brightness.light;
    case ThemeMode.system:
      return WidgetsBinding.instance.platformDispatcher.platformBrightness;
  }
}

class AppThemeScope extends InheritedWidget {
  const AppThemeScope({
    super.key,
    required this.themeMode,
    required this.platformBrightness,
    required this.toggleTheme,
    required super.child,
  });

  final ThemeMode themeMode;
  final Brightness platformBrightness;
  final VoidCallback toggleTheme;

  static AppThemeScope of(BuildContext context) {
    final scope = context.dependOnInheritedWidgetOfExactType<AppThemeScope>();
    assert(scope != null, 'AppThemeScope not found in context');
    return scope!;
  }

  @override
  bool updateShouldNotify(covariant AppThemeScope oldWidget) {
    return oldWidget.themeMode != themeMode || oldWidget.platformBrightness != platformBrightness;
  }
}

class ThemeModeToggleButton extends StatelessWidget {
  const ThemeModeToggleButton({super.key});

  @override
  Widget build(BuildContext context) {
    final scope = AppThemeScope.of(context);
    final isDark = effectiveThemeBrightness(scope.themeMode) == Brightness.dark;
    return IconButton(
      icon: Icon(isDark ? Icons.light_mode_rounded : Icons.dark_mode_rounded),
      tooltip: isDark ? 'Switch to light theme' : 'Switch to dark theme',
      onPressed: scope.toggleTheme,
    );
  }
}

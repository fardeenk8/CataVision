import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../app_theme_scope.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

Route<void> _resultScreenRoute({required List<int> imageBytes, required String filename}) {
  return PageRouteBuilder<void>(
    pageBuilder: (context, animation, secondaryAnimation) => ResultScreen(
      imageBytes: imageBytes,
      filename: filename,
    ),
    transitionsBuilder: (context, animation, secondaryAnimation, child) {
      final curved = CurvedAnimation(parent: animation, curve: Curves.easeOutCubic);
      return FadeTransition(
        opacity: curved,
        child: SlideTransition(
          position: Tween<Offset>(begin: const Offset(0, 0.04), end: Offset.zero).animate(curved),
          child: child,
        ),
      );
    },
    transitionDuration: const Duration(milliseconds: 280),
    reverseTransitionDuration: const Duration(milliseconds: 220),
  );
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _api = ApiService();
  bool _checkingHealth = false;
  bool _serverOk = false;
  String? _healthError;

  @override
  void initState() {
    super.initState();
    _checkHealth();
  }

  Future<void> _checkHealth() async {
    setState(() {
      _checkingHealth = true;
      _healthError = null;
    });
    final ok = await _api.healthCheck();
    setState(() {
      _checkingHealth = false;
      _serverOk = ok;
      if (!ok) _healthError = 'Cannot reach server. Check base URL and Wi‑Fi.';
    });
  }

  Future<void> _pickAndPredict(ImageSource source) async {
    final picker = ImagePicker();
    final xfile = await picker.pickImage(source: source, maxWidth: 1024, imageQuality: 90);
    if (xfile == null || !mounted) return;
    final bytes = await xfile.readAsBytes();
    if (!mounted) return;
    Navigator.of(context).push(_resultScreenRoute(imageBytes: bytes, filename: xfile.name));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: CustomScrollView(
          slivers: [
            SliverFillRemaining(
              hasScrollBody: false,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const SizedBox(height: 8),
                    const Row(
                      children: [
                        Spacer(),
                        ThemeModeToggleButton(),
                      ],
                    ),
                    const SizedBox(height: 24),
                    Text(
                      'Cataract Detection',
                      style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                            letterSpacing: -0.5,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Upload a photo of your eye for AI-powered screening',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                            color: Theme.of(context).colorScheme.onSurfaceVariant,
                          ),
                    ),
                    const SizedBox(height: 48),
                    if (_checkingHealth)
                      const Center(child: CircularProgressIndicator())
                    else if (_healthError != null) ...[
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.errorContainer,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          children: [
                            Icon(Icons.warning_amber_rounded,
                                color: Theme.of(context).colorScheme.onErrorContainer),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                _healthError!,
                                style: TextStyle(
                                  color: Theme.of(context).colorScheme.onErrorContainer,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                      FilledButton.tonalIcon(
                        onPressed: _checkHealth,
                        icon: const Icon(Icons.refresh),
                        label: const Text('Retry'),
                      ),
                    ] else ...[
                      _UploadCard(
                        icon: Icons.camera_alt_rounded,
                        label: 'Take photo',
                        onTap: () => _pickAndPredict(ImageSource.camera),
                      ),
                      const SizedBox(height: 16),
                      _UploadCard(
                        icon: Icons.photo_library_rounded,
                        label: 'Choose from gallery',
                        onTap: () => _pickAndPredict(ImageSource.gallery),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _UploadCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _UploadCard({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Theme.of(context).colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 20),
          child: Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: Theme.of(context).colorScheme.onPrimaryContainer, size: 26),
              ),
              const SizedBox(width: 16),
              Text(
                label,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
              ),
              const Spacer(),
              Icon(Icons.chevron_right_rounded, color: Theme.of(context).colorScheme.onSurfaceVariant),
            ],
          ),
        ),
      ),
    );
  }
}

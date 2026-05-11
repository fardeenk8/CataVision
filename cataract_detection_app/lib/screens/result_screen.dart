import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../app_theme_scope.dart';
import '../services/api_service.dart';

String _confidenceBandLabel(double confidence) {
  final c = confidence.clamp(0.0, 1.0);
  if (c >= 0.8) return 'High confidence';
  if (c >= 0.5) return 'Moderate confidence';
  return 'Low confidence';
}

class ResultScreen extends StatefulWidget {
  final List<int> imageBytes;
  final String filename;

  const ResultScreen({
    super.key,
    required this.imageBytes,
    required this.filename,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  final ApiService _api = ApiService();
  bool _loading = true;
  String? _error;
  PredictResult? _result;

  @override
  void initState() {
    super.initState();
    _predict();
  }

  Future<void> _predict() async {
    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });
    try {
      final result = await _api.predict(widget.imageBytes, widget.filename);
      if (mounted) setState(() => _result = result);
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _popToHome() {
    if (mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Result'),
        centerTitle: true,
        actions: const [
          ThemeModeToggleButton(),
        ],
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: SafeArea(
              bottom: false,
              child: _loading
                  ? _AnalyzingBody(imageBytes: widget.imageBytes)
                  : _error != null
                      ? _ErrorBody(message: _error!, onRetry: _predict)
                      : _result != null
                          ? _ResultBody(result: _result!)
                          : const SizedBox.shrink(),
            ),
          ),
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
              child: FilledButton.icon(
                onPressed: _popToHome,
                icon: const Icon(Icons.add_photo_alternate_rounded),
                label: const Text('Analyze another image'),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AnalyzingBody extends StatelessWidget {
  final List<int> imageBytes;

  const _AnalyzingBody({required this.imageBytes});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final surface = theme.colorScheme.surfaceContainerHighest;

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Analyzing…',
            style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            'Hold on while we screen your photo.',
            style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
          ),
          const SizedBox(height: 24),
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: AspectRatio(
              aspectRatio: 1,
              child: Image.memory(
                Uint8List.fromList(imageBytes),
                fit: BoxFit.cover,
                gaplessPlayback: true,
              ),
            ),
          ),
          const SizedBox(height: 24),
          SizedBox(
            height: 4,
            child: LinearProgressIndicator(
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          const SizedBox(height: 28),
          _SkeletonLine(color: surface, widthFactor: 1),
          const SizedBox(height: 12),
          _SkeletonLine(color: surface, widthFactor: 0.72),
          const SizedBox(height: 12),
          _SkeletonLine(color: surface, widthFactor: 0.88),
        ],
      ),
    );
  }
}

class _SkeletonLine extends StatelessWidget {
  final Color color;
  final double widthFactor;

  const _SkeletonLine({required this.color, required this.widthFactor});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: FractionallySizedBox(
        widthFactor: widthFactor,
        child: Container(
          height: 12,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(6),
          ),
        ),
      ),
    );
  }
}

class _ErrorBody extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ErrorBody({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.error_outline_rounded, size: 64, color: Theme.of(context).colorScheme.error),
          const SizedBox(height: 16),
          Text(
            message,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodyLarge,
          ),
          const SizedBox(height: 24),
          FilledButton.tonalIcon(onPressed: onRetry, icon: const Icon(Icons.refresh), label: const Text('Retry')),
        ],
      ),
    );
  }
}

class _ResultBody extends StatelessWidget {
  final PredictResult result;

  const _ResultBody({required this.result});

  @override
  Widget build(BuildContext context) {
    final isCataract = result.label == 'cataract';
    final theme = Theme.of(context);
    final confidence = result.confidence.clamp(0.0, 1.0);
    final bandLabel = _confidenceBandLabel(confidence);
    final onBanner = isCataract ? theme.colorScheme.onErrorContainer : theme.colorScheme.onPrimaryContainer;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            decoration: BoxDecoration(
              color: isCataract ? theme.colorScheme.errorContainer : theme.colorScheme.primaryContainer,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Row(
              children: [
                Icon(
                  isCataract ? Icons.visibility_off_rounded : Icons.check_circle_rounded,
                  size: 32,
                  color: onBanner,
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        result.label == 'cataract' ? 'Cataract detected' : 'No cataract',
                        style: theme.textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: onBanner,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        bandLabel,
                        style: theme.textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                          color: onBanner,
                        ),
                      ),
                      const SizedBox(height: 10),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: confidence,
                          minHeight: 8,
                          backgroundColor: onBanner.withValues(alpha: 0.25),
                          color: onBanner,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '${(confidence * 100).toStringAsFixed(1)}% model confidence',
                        style: theme.textTheme.bodySmall?.copyWith(color: onBanner.withValues(alpha: 0.9)),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          Text(
            'Grad-CAM overlay',
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                  fontWeight: FontWeight.w600,
                ),
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Container(
              decoration: BoxDecoration(
                border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Image.memory(
                base64Decode(result.gradcamBase64),
                fit: BoxFit.contain,
              ),
            ),
          ),
          const SizedBox(height: 24),
          DecoratedBox(
            decoration: BoxDecoration(
              color: theme.colorScheme.errorContainer.withValues(alpha: 0.35),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: theme.colorScheme.outlineVariant),
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.medical_information_outlined,
                    size: 22,
                    color: theme.colorScheme.onErrorContainer,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Not a medical diagnosis. This tool supports screening only—see a qualified clinician for diagnosis and care.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurface,
                        height: 1.35,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

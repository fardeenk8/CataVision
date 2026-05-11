import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config.dart';

class PredictResult {
  final String label;
  final double confidence;
  final double probCataract;
  final String gradcamBase64;

  PredictResult({
    required this.label,
    required this.confidence,
    required this.probCataract,
    required this.gradcamBase64,
  });

  factory PredictResult.fromJson(Map<String, dynamic> json) {
    return PredictResult(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      probCataract: (json['prob_cataract'] as num).toDouble(),
      gradcamBase64: json['gradcam_b64'] as String,
    );
  }
}

class ApiService {
  static final ApiService _instance = ApiService._();
  factory ApiService() => _instance;

  ApiService._();

  String get baseUrl => kApiBaseUrl.endsWith('/') ? kApiBaseUrl : '$kApiBaseUrl';

  Future<bool> healthCheck() async {
    try {
      final r = await http.get(Uri.parse('$baseUrl/health')).timeout(
        const Duration(seconds: 5),
      );
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<PredictResult> predict(List<int> imageBytes, String filename) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes(
      'image',
      imageBytes,
      filename: filename.isEmpty ? 'image.jpg' : filename,
    ));
    if (kApiKey != null && kApiKey!.isNotEmpty) {
      request.headers['X-API-Key'] = kApiKey!;
    }
    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode != 200) {
      final body = response.body;
      String msg = 'Request failed';
      try {
        final j = jsonDecode(body) as Map<String, dynamic>;
        if (j.containsKey('error')) msg = j['error'] as String;
      } catch (_) {}
      throw ApiException(msg, response.statusCode);
    }
    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return PredictResult.fromJson(json);
  }
}

class ApiException implements Exception {
  final String message;
  final int? statusCode;
  ApiException(this.message, [this.statusCode]);
  @override
  String toString() => message;
}

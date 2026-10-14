import 'dart:convert';

import 'package:http/http.dart' as http;

class ApiException implements Exception {
  const ApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

class PatientSummary {
  const PatientSummary({required this.id, required this.label});

  final String id;
  final String label;

  factory PatientSummary.fromJson(Map<String, dynamic> json) {
    return PatientSummary(
      id: json['id'] as String,
      label: json['label'] as String,
    );
  }
}

class HealthStatus {
  const HealthStatus({
    required this.status,
    required this.modelReady,
    required this.patientCount,
  });

  final String status;
  final bool modelReady;
  final int patientCount;

  factory HealthStatus.fromJson(Map<String, dynamic> json) {
    return HealthStatus(
      status: json['status'] as String,
      modelReady: json['modelReady'] as bool,
      patientCount: json['patientCount'] as int,
    );
  }
}

class ChatHistoryMessage {
  const ChatHistoryMessage({required this.role, required this.content});

  final String role;
  final String content;

  Map<String, String> toJson() => {'role': role, 'content': content};
}

class ChatResult {
  const ChatResult({
    required this.answer,
    required this.model,
    required this.latencySeconds,
  });

  final String answer;
  final String model;
  final double latencySeconds;

  factory ChatResult.fromJson(Map<String, dynamic> json) {
    return ChatResult(
      answer: json['answer'] as String,
      model: json['model'] as String,
      latencySeconds: (json['latencyMs'] as num).toDouble() / 1000,
    );
  }
}

class MobileApiClient {
  MobileApiClient({
    required String baseUrl,
    required String token,
    http.Client? httpClient,
  }) : _baseUri = _parseBaseUrl(baseUrl),
       _token = _validateToken(token),
       _client = httpClient ?? http.Client();

  final Uri _baseUri;
  final String _token;
  final http.Client _client;

  static Uri _parseBaseUrl(String value) {
    final normalized = value.trim().replaceFirst(RegExp(r'/+$'), '');
    final uri = Uri.tryParse(normalized);
    if (uri == null ||
        !uri.hasAuthority ||
        (uri.scheme != 'http' && uri.scheme != 'https')) {
      throw const ApiException('Geçerli bir http veya https API adresi girin.');
    }
    return uri;
  }

  static String _validateToken(String value) {
    final token = value.trim();
    if (token.isEmpty) {
      throw const ApiException('Erişim kodu boş bırakılamaz.');
    }
    return token;
  }

  Map<String, String> get _headers => {
    'Accept': 'application/json',
    'Content-Type': 'application/json; charset=utf-8',
    'X-Demo-Token': _token,
  };

  Uri _uri(String path) => _baseUri.replace(path: path);

  Future<HealthStatus> health() async {
    final response = await _client
        .get(_uri('/api/health'), headers: _headers)
        .timeout(const Duration(seconds: 8));
    return HealthStatus.fromJson(_decodeObject(response));
  }

  Future<List<PatientSummary>> patients() async {
    final response = await _client
        .get(_uri('/api/patients'), headers: _headers)
        .timeout(const Duration(seconds: 8));
    final items = _decodeList(response);
    return items
        .map((item) => PatientSummary.fromJson(item as Map<String, dynamic>))
        .toList(growable: false);
  }

  Future<ChatResult> chat({
    required String patientId,
    required String question,
    required List<ChatHistoryMessage> history,
  }) async {
    final response = await _client
        .post(
          _uri('/api/chat'),
          headers: _headers,
          body: jsonEncode({
            'patientId': patientId,
            'question': question,
            'history': history.map((message) => message.toJson()).toList(),
          }),
        )
        .timeout(const Duration(seconds: 190));
    return ChatResult.fromJson(_decodeObject(response));
  }

  Object? _decode(http.Response response) {
    Object? body;
    try {
      body = jsonDecode(utf8.decode(response.bodyBytes));
    } on Object {
      if (response.statusCode >= 200 && response.statusCode < 300) {
        throw const ApiException('Sunucu geçersiz bir yanıt döndürdü.');
      }
    }

    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = body is Map<String, dynamic> ? body['detail'] : null;
      throw ApiException(
        detail is String ? detail : 'Sunucu isteği tamamlayamadı.',
        statusCode: response.statusCode,
      );
    }
    if (body == null) {
      throw const ApiException('Sunucu yanıtı okunamadı.');
    }
    return body;
  }

  Map<String, dynamic> _decodeObject(http.Response response) {
    final body = _decode(response);
    if (body is! Map<String, dynamic>) {
      throw const ApiException('Sunucu yanıtı beklenen biçimde değil.');
    }
    return body;
  }

  List<dynamic> _decodeList(http.Response response) {
    final body = _decode(response);
    if (body is! List<dynamic>) {
      throw const ApiException('Sunucu yanıtı beklenen biçimde değil.');
    }
    return body;
  }

  void close() => _client.close();
}

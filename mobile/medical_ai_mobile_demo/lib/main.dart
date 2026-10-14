import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import 'api.dart';

void main() {
  runApp(const MedicalAiApp());
}

class MedicalAiApp extends StatelessWidget {
  const MedicalAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF1F7A5A),
      brightness: Brightness.light,
    );
    return MaterialApp(
      title: 'MedicalAI Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: colorScheme,
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF5F8F6),
        inputDecorationTheme: const InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.all(Radius.circular(14)),
          ),
        ),
      ),
      home: const ConnectionPage(),
    );
  }
}

class ConnectionPage extends StatefulWidget {
  const ConnectionPage({super.key});

  @override
  State<ConnectionPage> createState() => _ConnectionPageState();
}

class _ConnectionPageState extends State<ConnectionPage> {
  final _urlController = TextEditingController(text: 'http://10.34.14.76:8093');
  final _tokenController = TextEditingController();
  bool _obscureToken = true;
  bool _connecting = false;
  String? _error;

  @override
  void dispose() {
    _urlController.dispose();
    _tokenController.dispose();
    super.dispose();
  }

  Future<void> _connect() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _connecting = true;
      _error = null;
    });

    MobileApiClient? candidate;
    try {
      candidate = MobileApiClient(
        baseUrl: _urlController.text,
        token: _tokenController.text,
      );
      final results = await Future.wait([
        candidate.health(),
        candidate.patients(),
      ]);
      final health = results[0] as HealthStatus;
      final patients = results[1] as List<PatientSummary>;
      if (health.status != 'ready' || !health.modelReady || patients.isEmpty) {
        throw const ApiException(
          'Model veya hasta kayıtları henüz hazır değil.',
        );
      }
      if (!mounted) return;
      final connectedClient = candidate;
      candidate = null;
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => ChatPage(api: connectedClient, patients: patients),
        ),
      );
    } on TimeoutException {
      if (mounted) {
        setState(() {
          _error =
              'Sunucuya ulaşılamadı. Telefon ve bilgisayarın ofis '
              'WiFi ağında olduğunu kontrol edin.';
        });
      }
    } on ApiException catch (error) {
      if (mounted) setState(() => _error = error.message);
    } on Object {
      if (mounted) {
        setState(() {
          _error =
              'Bağlantı kurulamadı. API adresini ve erişim kodunu '
              'kontrol edin.';
        });
      }
    } finally {
      candidate?.close();
      if (mounted) setState(() => _connecting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 480),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Icon(
                    Icons.health_and_safety_outlined,
                    size: 64,
                    color: Color(0xFF1F7A5A),
                  ),
                  const SizedBox(height: 18),
                  Text(
                    'MedicalAI',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Kişisel sağlık sohbeti',
                    textAlign: TextAlign.center,
                    style: Theme.of(
                      context,
                    ).textTheme.titleMedium?.copyWith(color: Colors.black54),
                  ),
                  const SizedBox(height: 32),
                  TextField(
                    controller: _urlController,
                    keyboardType: TextInputType.url,
                    autocorrect: false,
                    decoration: const InputDecoration(
                      labelText: 'API adresi',
                      prefixIcon: Icon(Icons.lan_outlined),
                    ),
                  ),
                  const SizedBox(height: 14),
                  TextField(
                    controller: _tokenController,
                    obscureText: _obscureToken,
                    autocorrect: false,
                    enableSuggestions: false,
                    onSubmitted: (_) => _connecting ? null : _connect(),
                    decoration: InputDecoration(
                      labelText: 'Erişim kodu',
                      prefixIcon: const Icon(Icons.key_outlined),
                      suffixIcon: IconButton(
                        onPressed: () {
                          setState(() => _obscureToken = !_obscureToken);
                        },
                        icon: Icon(
                          _obscureToken
                              ? Icons.visibility_outlined
                              : Icons.visibility_off_outlined,
                        ),
                      ),
                    ),
                  ),
                  if (_error != null) ...[
                    const SizedBox(height: 14),
                    _StatusCard(message: _error!),
                  ],
                  const SizedBox(height: 18),
                  FilledButton.icon(
                    onPressed: _connecting ? null : _connect,
                    icon: _connecting
                        ? const SizedBox.square(
                            dimension: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.login),
                    label: Text(
                      _connecting ? 'Bağlanıyor…' : 'Sunucuya bağlan',
                    ),
                  ),
                  const SizedBox(height: 20),
                  const _InfoRow(
                    icon: Icons.wifi,
                    text: 'Erişim yalnızca ofis WiFi ağı içinden açıktır.',
                  ),
                  const SizedBox(height: 10),
                  const _InfoRow(
                    icon: Icons.lock_outline,
                    text:
                        'Hasta kayıtları telefona indirilmez; sohbet geçmişi '
                        'yalnızca bu oturumda bellekte tutulur.',
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _StatusCard extends StatelessWidget {
  const _StatusCard({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.errorContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        message,
        style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  const _InfoRow({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20, color: const Color(0xFF1F7A5A)),
        const SizedBox(width: 10),
        Expanded(child: Text(text, style: const TextStyle(height: 1.35))),
      ],
    );
  }
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key, required this.api, required this.patients});

  final MobileApiClient api;
  final List<PatientSummary> patients;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  static const _starterQuestions = [
    'Son zamanlarda çok yorgunum, neden olabilir?',
    'Başım ağrıyor, kayıtlarımda bununla ilgili bir şey var mı?',
    'Gözüm biraz kızarık, neden olabilir?',
    'Uykum neden kalitesiz olabilir?',
    'Bazen çarpıntı hissediyorum, neye dikkat etmeliyim?',
  ];

  final _questionController = TextEditingController();
  final _scrollController = ScrollController();
  final List<_ChatMessage> _messages = [];
  late String _patientId;
  bool _waiting = false;

  @override
  void initState() {
    super.initState();
    _patientId = widget.patients.first.id;
  }

  @override
  void dispose() {
    widget.api.close();
    _questionController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _resetConversation() {
    setState(() {
      _messages.clear();
      _questionController.clear();
    });
  }

  Future<void> _send([String? suggestedQuestion]) async {
    if (_waiting) return;
    final question = (suggestedQuestion ?? _questionController.text).trim();
    if (question.isEmpty) return;
    FocusScope.of(context).unfocus();

    final history = _messages
        .where((message) => message.error == null)
        .map(
          (message) => ChatHistoryMessage(
            role: message.fromUser ? 'user' : 'assistant',
            content: message.content,
          ),
        )
        .toList(growable: false);
    if (history.length > 6) {
      history.removeRange(0, history.length - 6);
    }

    setState(() {
      _messages.add(_ChatMessage.user(question));
      _questionController.clear();
      _waiting = true;
    });
    _scrollToBottom();

    try {
      final result = await widget.api.chat(
        patientId: _patientId,
        question: question,
        history: history,
      );
      if (!mounted) return;
      setState(() {
        _messages.add(
          _ChatMessage.assistant(
            result.answer,
            latencySeconds: result.latencySeconds,
            model: result.model,
          ),
        );
      });
    } on TimeoutException {
      if (mounted) {
        setState(() {
          _messages.add(
            _ChatMessage.failure(
              'Yanıt süresi aşıldı. Sunucuyu kontrol edip tekrar deneyin.',
            ),
          );
        });
      }
    } on ApiException catch (error) {
      if (mounted) {
        setState(() => _messages.add(_ChatMessage.failure(error.message)));
      }
    } on Object {
      if (mounted) {
        setState(() {
          _messages.add(
            _ChatMessage.failure('Yanıt alınamadı. Bağlantıyı kontrol edin.'),
          );
        });
      }
    } finally {
      if (mounted) {
        setState(() => _waiting = false);
        _scrollToBottom();
      }
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('MedicalAI'),
        actions: [
          IconButton(
            tooltip: 'Sohbeti sıfırla',
            onPressed: _messages.isEmpty ? null : _resetConversation,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            _PatientSelector(
              patients: widget.patients,
              value: _patientId,
              enabled: !_waiting,
              onChanged: (value) {
                if (value == null || value == _patientId) return;
                setState(() {
                  _patientId = value;
                  _messages.clear();
                });
              },
            ),
            Expanded(
              child: _messages.isEmpty
                  ? _EmptyConversation(
                      questions: _starterQuestions,
                      onSelected: _send,
                    )
                  : ListView.builder(
                      controller: _scrollController,
                      padding: const EdgeInsets.fromLTRB(16, 12, 16, 18),
                      itemCount: _messages.length + (_waiting ? 1 : 0),
                      itemBuilder: (context, index) {
                        if (_waiting && index == _messages.length) {
                          return const _ThinkingBubble();
                        }
                        return _MessageBubble(
                          message: _messages[index],
                          onQuickReply: _send,
                        );
                      },
                    ),
            ),
            _Composer(
              controller: _questionController,
              enabled: !_waiting,
              onSend: () => _send(),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 6, 16, 10),
              child: Text(
                'Bu demo tıbbi tanı koymaz. Acil bir durumda 112’yi arayın.',
                textAlign: TextAlign.center,
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: Colors.black54),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PatientSelector extends StatelessWidget {
  const _PatientSelector({
    required this.patients,
    required this.value,
    required this.enabled,
    required this.onChanged,
  });

  final List<PatientSummary> patients;
  final String value;
  final bool enabled;
  final ValueChanged<String?> onChanged;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
        child: DropdownButtonFormField<String>(
          initialValue: value,
          decoration: const InputDecoration(
            labelText: 'Hasta profili',
            prefixIcon: Icon(Icons.person_outline),
            isDense: true,
          ),
          items: patients
              .map(
                (patient) => DropdownMenuItem(
                  value: patient.id,
                  child: Text('${patient.label} (${patient.id})'),
                ),
              )
              .toList(growable: false),
          onChanged: enabled ? onChanged : null,
        ),
      ),
    );
  }
}

class _EmptyConversation extends StatelessWidget {
  const _EmptyConversation({required this.questions, required this.onSelected});

  final List<String> questions;
  final ValueChanged<String> onSelected;

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        const SizedBox(height: 16),
        Text(
          'Bugün nasıl hissediyorsun?',
          textAlign: TextAlign.center,
          style: Theme.of(
            context,
          ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 8),
        const Text(
          'Kendi sorunuzu yazabilir veya aşağıdaki örneklerden biriyle '
          'başlayabilirsiniz.',
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 22),
        for (final question in questions)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: OutlinedButton(
              onPressed: () => onSelected(question),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.all(14),
                alignment: Alignment.centerLeft,
              ),
              child: Text(question),
            ),
          ),
      ],
    );
  }
}

class _ChatMessage {
  const _ChatMessage({
    required this.content,
    required this.fromUser,
    this.latencySeconds,
    this.model,
    this.error,
  });

  factory _ChatMessage.user(String content) {
    return _ChatMessage(content: content, fromUser: true);
  }

  factory _ChatMessage.assistant(
    String content, {
    required double latencySeconds,
    required String model,
  }) {
    return _ChatMessage(
      content: content,
      fromUser: false,
      latencySeconds: latencySeconds,
      model: model,
    );
  }

  factory _ChatMessage.failure(String message) {
    return _ChatMessage(content: message, fromUser: false, error: message);
  }

  final String content;
  final bool fromUser;
  final double? latencySeconds;
  final String? model;
  final String? error;
}

class _MessageBubble extends StatelessWidget {
  const _MessageBubble({required this.message, required this.onQuickReply});

  final _ChatMessage message;
  final ValueChanged<String> onQuickReply;

  static final _optionPattern = RegExp(
    r'^\s*(?:[-*]\s*)?([A-E])\)\s+(.+?)\s*$',
    multiLine: true,
  );

  List<String> get _quickReplies => _optionPattern
      .allMatches(message.content)
      .map((match) => match.group(2)!.trim())
      .toSet()
      .toList(growable: false);

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final quickReplies = message.fromUser ? const <String>[] : _quickReplies;
    final background = message.error != null
        ? colors.errorContainer
        : message.fromUser
        ? colors.primary
        : Colors.white;
    final foreground = message.error != null
        ? colors.onErrorContainer
        : message.fromUser
        ? colors.onPrimary
        : colors.onSurface;

    return Align(
      alignment: message.fromUser
          ? Alignment.centerRight
          : Alignment.centerLeft,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 620),
        margin: EdgeInsets.only(
          bottom: 12,
          left: message.fromUser ? 42 : 0,
          right: message.fromUser ? 0 : 22,
        ),
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 10),
        decoration: BoxDecoration(
          color: background,
          borderRadius: BorderRadius.circular(16),
          boxShadow: message.fromUser
              ? null
              : const [
                  BoxShadow(
                    color: Color(0x12000000),
                    blurRadius: 8,
                    offset: Offset(0, 2),
                  ),
                ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (message.fromUser)
              Text(message.content, style: TextStyle(color: foreground))
            else
              MarkdownBody(
                data: message.content,
                selectable: true,
                styleSheet: MarkdownStyleSheet(
                  p: TextStyle(color: foreground, height: 1.42),
                  strong: TextStyle(
                    color: foreground,
                    fontWeight: FontWeight.w700,
                  ),
                  listBullet: TextStyle(color: foreground),
                ),
              ),
            if (quickReplies.isNotEmpty) ...[
              const SizedBox(height: 8),
              Wrap(
                spacing: 7,
                runSpacing: 7,
                children: quickReplies
                    .map(
                      (reply) => ActionChip(
                        label: Text(reply),
                        onPressed: () => onQuickReply(reply),
                      ),
                    )
                    .toList(growable: false),
              ),
            ],
            if (message.latencySeconds != null) ...[
              const SizedBox(height: 8),
              Text(
                '${message.latencySeconds!.toStringAsFixed(1)} sn • '
                '${message.model}',
                style: Theme.of(
                  context,
                ).textTheme.labelSmall?.copyWith(color: Colors.black45),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ThinkingBubble extends StatelessWidget {
  const _ThinkingBubble();

  @override
  Widget build(BuildContext context) {
    return const Align(
      alignment: Alignment.centerLeft,
      child: Padding(
        padding: EdgeInsets.only(bottom: 12),
        child: Chip(
          avatar: SizedBox.square(
            dimension: 15,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          label: Text('Kayıtlar değerlendiriliyor…'),
        ),
      ),
    );
  }
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.controller,
    required this.enabled,
    required this.onSend,
  });

  final TextEditingController controller;
  final bool enabled;
  final VoidCallback onSend;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      elevation: 8,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 10, 10, 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Expanded(
              child: TextField(
                controller: controller,
                enabled: enabled,
                minLines: 1,
                maxLines: 5,
                textCapitalization: TextCapitalization.sentences,
                decoration: const InputDecoration(
                  hintText: 'Sorunuzu yazın…',
                  border: InputBorder.none,
                  filled: false,
                ),
              ),
            ),
            const SizedBox(width: 6),
            IconButton.filled(
              tooltip: 'Gönder',
              onPressed: enabled ? onSend : null,
              icon: const Icon(Icons.arrow_upward),
            ),
          ],
        ),
      ),
    );
  }
}

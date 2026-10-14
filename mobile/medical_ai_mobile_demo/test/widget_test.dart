import 'package:flutter_test/flutter_test.dart';
import 'package:medical_ai_mobile_demo/main.dart';

void main() {
  testWidgets('connection screen exposes office demo settings', (tester) async {
    await tester.pumpWidget(const MedicalAiApp());

    expect(find.text('MedicalAI'), findsOneWidget);
    expect(find.text('Kişisel sağlık sohbeti'), findsOneWidget);
    expect(find.text('API adresi'), findsOneWidget);
    expect(find.text('Erişim kodu'), findsOneWidget);
    expect(find.text('Sunucuya bağlan'), findsOneWidget);
    expect(find.textContaining('ofis WiFi'), findsOneWidget);
  });
}

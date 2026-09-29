import 'package:flutter_test/flutter_test.dart';
import 'package:stetho_ai/main.dart';

void main() {
  testWidgets('StethoAiApp smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const StethoAiApp());
    expect(find.text('StethoAI'), findsWidgets);
  });
}

# MedicalAI Android demo

Bu istemci, mevcut üç hasta profilini ofis WiFi ağı içindeki ayrı demo API'si
üzerinden Gemma'ya gönderir. Uygulamada sabit klinik yanıt yoktur; başlangıç
soruları yalnızca kullanıcıya örnektir. API, hasta CCO metnini telefona dönmez.

## Çalıştırma

Komutları repo kökünde PowerShell ile çalıştırın.

1. Windows Güvenlik Duvarı kuralını yönetici PowerShell'de bir kez oluşturun:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\configure_mobile_demo_firewall.ps1
   ```

2. GPU servisinin yerel tünelini ayrı bir terminalde açın:

   ```powershell
   ssh -N -L 18000:127.0.0.1:8000 arge-adm@185.48.24.103
   ```

3. API'yi başlatın:

   ```powershell
   .\scripts\start_mobile_chat_api.ps1
   ```

   Script, telefon uygulamasına girilecek API adresini ve o çalıştırmaya özel
   erişim kodunu ekrana yazar. Bilgisayar ofis ağı olan `10.34.14.0/23` içinde
   değilse API başlamaz.

4. Android cihazda USB hata ayıklamayı açıp cihazı bağlayın:

   ```powershell
   cd .\mobile\medical_ai_mobile_demo
   flutter devices
   flutter run
   ```

   Alternatif olarak `flutter build apk --debug` ile üretilen debug APK'yı
   telefona kurabilirsiniz.

5. Telefonda bilgisayarla aynı ofis WiFi ağına bağlanın. Uygulamada API adresi
   olarak `http://10.34.14.76:8093` ve API terminalinde gösterilen erişim kodunu
   kullanın.

## Güvenlik sınırı

- API yalnızca `10.34.14.0/23` ve localhost istemcilerini kabul eder.
- Windows Güvenlik Duvarı port 8093'ü sadece aynı ofis CIDR'ına açar.
- Token uygulamada kalıcı depolamaya yazılmaz; yalnızca bellekte tutulur.
- HTTP izni sadece Android debug manifestindedir. Release derlemesi için HTTPS
  kurulmadan uygulama ofis dışına çıkarılmamalıdır.
- Bu yapı production servisini veya mevcut web demosunu değiştirmez.

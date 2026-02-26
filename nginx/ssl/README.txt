SSL-сертификаты для HTTPS

Положите сюда:
  - cert.pem  (сертификат)
  - key.pem   (приватный ключ)

Самоподписанный сертификат для локальной сети (Chrome покажет предупреждение — можно принять):
  ./gen-selfsigned.sh

Или вручную:
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout key.pem -out cert.pem -subj "/CN=localhost"

Для доступа по имени хоста замените CN=localhost на CN=your-hostname.local

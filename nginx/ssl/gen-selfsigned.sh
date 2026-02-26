#!/bin/bash
# Генерация самоподписанного сертификата для HTTPS в локальной сети
cd "$(dirname "$0")"
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem -subj "/CN=localhost"
echo "Созданы cert.pem и key.pem"

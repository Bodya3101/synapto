import os
import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Dict, Any

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class SecurityError(Exception):
    """Исключение при нарушении векторов безопасности библиотеки."""
    pass


class SafetyUtils:
    @staticmethod
    def validate_and_sanitize_path(filepath: str) -> Path:
        if not isinstance(filepath, str) or not filepath.strip():
            raise SecurityError("Путь к файлу должен быть непустой строкой.")
            
        path = Path(filepath).resolve()
        if path.suffix not in [".safetensors", ".json", ".enc"]:
            raise SecurityError("Разрешена работа только с безопасными расширениями (.safetensors, .json, .enc).")
            
        return path

    @staticmethod
    def sanitize_input_text(text: str, max_chars: int = 4096) -> str:
        if not isinstance(text, str):
            raise TypeError("Входные данные должны быть строкового типа.")
        if len(text) > max_chars:
            return text[:max_chars]
        return text


class CryptoVault:
    """
    Модуль E2E-шифрования метаданных памяти (PBKDF2 + AES-256-CBC с HMAC-SHA256 подписью).
    """
    SYSTEM_PEPPER = b"SYNAPTO_E2E_SYSTEM_PEPPER_V1_SECRET_KEY"

    @classmethod
    def _derive_keys(cls, master_key: str, salt: bytes) -> tuple[bytes, bytes]:
        """
        Генерирует 256-битный ключ шифрования и 256-битный ключ HMAC из пароля, соли и перца.
        """
        key_bytes = master_key.encode("utf-8")
        peppered_key = hmac.new(cls.SYSTEM_PEPPER, key_bytes, hashlib.sha256).digest()
        
        # 512-битный вывод: первые 32 байта — AES ключ, следующие 32 байта — HMAC ключ
        derived = hashlib.pbkdf2_hmac("sha256", peppered_key, salt, 100000, dklen=64)
        return derived[:32], derived[32:]

    @classmethod
    def encrypt_metadata(cls, data: Dict[str, Any], master_key: str) -> bytes:
        json_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        salt = os.urandom(16)
        iv = os.urandom(16)
        enc_key, mac_key = cls._derive_keys(master_key, salt)

        if HAS_CRYPTOGRAPHY:
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(json_bytes) + padder.finalize()
            cipher = Cipher(algorithms.AES(enc_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        else:
            # Резервное шифрование через HMAC-Stream Cipher (если cryptography не установлена)
            ciphertext = bytearray()
            for i, byte in enumerate(json_bytes):
                ks = hmac.new(enc_key, iv + (i // 32).to_bytes(4, 'big'), hashlib.sha256).digest()
                ciphertext.append(byte ^ ks[i % 32])
            ciphertext = bytes(ciphertext)

        payload = salt + iv + ciphertext
        mac = hmac.new(mac_key, payload, hashlib.sha256).digest()
        return base64.b64encode(mac + payload)

    @classmethod
    def decrypt_metadata(cls, encrypted_base64: bytes, master_key: str) -> Dict[str, Any]:
        try:
            raw_pack = base64.b64decode(encrypted_base64)
            mac = raw_pack[:32]
            payload = raw_pack[32:]
            
            salt = payload[:16]
            iv = payload[16:32]
            ciphertext = payload[32:]

            enc_key, mac_key = cls._derive_keys(master_key, salt)

            # Проверка подлинности HMAC
            expected_mac = hmac.new(mac_key, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected_mac):
                raise SecurityError("Аутентификация не пройдена: данные повреждены или подделай ключ.")

            if HAS_CRYPTOGRAPHY:
                cipher = Cipher(algorithms.AES(enc_key), modes.CBC(iv), backend=default_backend())
                decryptor = cipher.decryptor()
                padded_data = decryptor.update(ciphertext) + decryptor.finalize()
                unpadder = padding.PKCS7(128).unpadder()
                json_bytes = unpadder.update(padded_data) + unpadder.finalize()
            else:
                json_bytes = bytearray()
                for i, byte in enumerate(ciphertext):
                    ks = hmac.new(enc_key, iv + (i // 32).to_bytes(4, 'big'), hashlib.sha256).digest()
                    json_bytes.append(byte ^ ks[i % 32])
                json_bytes = bytes(json_bytes)

            return json.loads(json_bytes.decode("utf-8"))
        except Exception as e:
            raise SecurityError(f"Неверный ключ шифрования или поврежденный файл: {str(e)}")
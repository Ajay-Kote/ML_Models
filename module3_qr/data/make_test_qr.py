"""
make_test_qr.py
----------------
Quick helper to generate a single custom QR code (legit or malicious-styled)
for manually testing predict.py.

Usage:
    python make_test_qr.py "http://192.168.1.5/paypal-verify-login" fake_test.png --malicious
    python make_test_qr.py "https://github.com" real_test.png
"""
import sys
import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_Q

def main():
    if len(sys.argv) < 3:
        print('Usage: python make_test_qr.py "<url>" <output.png> [--malicious]')
        sys.exit(1)

    url = sys.argv[1]
    out_path = sys.argv[2]
    malicious = "--malicious" in sys.argv

    # Malicious-styled: low error correction, small box, tight border
    # (mirrors the same generation pattern used to build the training set)
    ec = ERROR_CORRECT_L if malicious else ERROR_CORRECT_Q
    box_size = 5 if malicious else 8
    border = 1 if malicious else 4

    qr = qrcode.QRCode(error_correction=ec, box_size=box_size, border=border)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    img.save(out_path)
    print(f"Saved QR for '{url}' -> {out_path}  (style: {'malicious' if malicious else 'legit'})")

if __name__ == "__main__":
    main()

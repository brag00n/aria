"""HTTPS d'Aria (réglages : section "https" de la config).

Les navigateurs n'autorisent le micro que sur une page sécurisée (HTTPS ou localhost) : depuis un autre poste ou un
smartphone, http://IP:7860 affiche l'interface mais le micro est refusé. Avec "https" activé, Aria génère un certificat
auto-signé (une seule fois), puis lance le serveur dessus. Le navigateur affiche un avertissement à accepter une fois.

Le certificat est stocké dans models/certs/ (dossier ignoré par git, monté dans le conteneur) et jamais commité.
Module isolé : app.py n'a qu'à appeler launch_args().
"""
import datetime
import ipaddress
import os
import components.utils as utils

CERT_DIR = utils.aria_path("models", "certs")
CERT_FILE = os.path.join(CERT_DIR, "aria.crt")
KEY_FILE = os.path.join(CERT_DIR, "aria.key")
VALID_DAYS = 825   # durée maximale acceptée par les navigateurs pour un certificat


def _wanted_names(params):
    """Noms et adresses que le certificat doit couvrir : localhost + ceux de la config (ex. l'IP du PC)."""
    names = ["localhost", "127.0.0.1"]
    for item in (params.get("hosts") or []):
        item = str(item).strip()
        if item and item not in names:
            names.append(item)
    return names


def _generate(names):
    """Crée une clé et un certificat auto-signé couvrant les noms demandés."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Aria")])
    sans = []
    for n in names:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            sans.append(x509.DNSName(n))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=VALID_DAYS))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))

    os.makedirs(CERT_DIR, exist_ok=True)
    with open(KEY_FILE, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    try:
        os.chmod(KEY_FILE, 0o600)
    except OSError:
        pass
    with open(CERT_FILE, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(os.path.join(CERT_DIR, "aria.names"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    utils.log_info("HTTPS", f"🔐 Certificat auto-signé créé pour : {', '.join(names)}")


def _is_current(names):
    """Le certificat existe, couvre les noms demandés et n'expire pas dans les 30 jours."""
    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        return False
    try:
        with open(os.path.join(CERT_DIR, "aria.names"), encoding="utf-8") as f:
            if set(f.read().split()) != set(names):
                utils.log_info("HTTPS", "Noms du certificat modifiés : régénération")
                return False
        from cryptography import x509
        with open(CERT_FILE, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())
        left = cert.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)
        if left < datetime.timedelta(days=30):
            utils.log_info("HTTPS", "Certificat bientôt expiré : régénération")
            return False
        return True
    except Exception as e:
        utils.log_info("HTTPS", f"Certificat illisible ({e}) : régénération")
        return False


def launch_args(params):
    """Arguments à ajouter à launch() : certificat et clé si le HTTPS est activé, sinon rien (HTTP comme avant).
    En cas d'échec de création du certificat, Aria démarre en HTTP au lieu de ne pas démarrer."""
    # Tags <NOM> -> variables du .env (ex. "hosts": ["<IP_ADDRESS>"]) : l'adresse du PC n'est pas écrite dans la config
    params = utils.expand_secrets(params or {}, "https")
    if not params.get("enabled", False):
        return {}
    names = _wanted_names(params)
    try:
        if not _is_current(names):
            _generate(names)
        utils.log_info("HTTPS", f"HTTPS activé (certificat pour : {', '.join(names)})")
        return {"ssl_certfile": CERT_FILE, "ssl_keyfile": KEY_FILE, "ssl_verify": False}
    except Exception as e:
        utils.log_info("HTTPS", f"❌ Certificat impossible ({type(e).__name__}: {e}) : démarrage en HTTP")
        return {}

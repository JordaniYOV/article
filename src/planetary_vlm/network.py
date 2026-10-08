"""Verified HTTPS with system roots, optional certifi and an explicit CA bundle."""

import os
import ssl
import urllib.error
import urllib.request


def create_ssl_context():
    """Keep system trust and hostname verification; never use an insecure fallback."""
    context = ssl.create_default_context()
    bundle = os.environ.get("SSL_CERT_FILE")
    if not bundle:
        try:
            import certifi
        except ImportError:
            return context
        bundle = certifi.where()
    # Explicit loading also catches an invalid SSL_CERT_FILE instead of silently
    # ignoring it. Extra roots augment rather than replace the system store.
    context.load_verify_locations(cafile=bundle)
    return context


def certificate_error(error):
    return (
        f"HTTPS certificate verification failed: {error}. "
        "Install/update certifi in the API Python environment. "
        "If your network uses a proxy or HTTPS inspection, set SSL_CERT_FILE "
        "to its trusted PEM CA bundle and restart the API."
    )


def urlopen(request, *, timeout=45):
    try:
        return urllib.request.urlopen(request, timeout=timeout, context=create_ssl_context())
    except urllib.error.URLError as error:
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            raise urllib.error.URLError(certificate_error(error.reason)) from error
        raise

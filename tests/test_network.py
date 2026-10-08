"""Certificate failures must be fixed with trust roots, not disabled verification."""

import os
import ssl
import sys
import types
import unittest
import urllib.error
from unittest.mock import Mock, patch
from io import BytesIO

from planetary_vlm import network


class NetworkTests(unittest.TestCase):
    def test_default_context_still_verifies_certificates_and_hostnames(self):
        with patch.dict(os.environ, {"SSL_CERT_FILE": ""}):
            context = network.create_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertGreater(context.cert_store_stats()["x509_ca"], 0)

    def test_certifi_roots_augment_system_context(self):
        context = Mock()
        certifi = types.SimpleNamespace(where=lambda: "mozilla-ca.pem")
        with patch.dict(os.environ, {"SSL_CERT_FILE": ""}), \
                patch.dict(sys.modules, {"certifi": certifi}), \
                patch.object(network.ssl, "create_default_context", return_value=context) as factory:
            self.assertIs(network.create_ssl_context(), context)
        factory.assert_called_once_with()
        context.load_verify_locations.assert_called_once_with(cafile="mozilla-ca.pem")

    def test_explicit_ca_bundle_is_loaded_without_certifi(self):
        context = Mock()
        with patch.dict(os.environ, {"SSL_CERT_FILE": "network-ca.pem"}), \
                patch.dict(sys.modules, {"certifi": None}), \
                patch.object(network.ssl, "create_default_context", return_value=context):
            network.create_ssl_context()
        context.load_verify_locations.assert_called_once_with(cafile="network-ca.pem")

    def test_missing_certifi_keeps_system_verification(self):
        with patch.dict(os.environ, {"SSL_CERT_FILE": ""}), patch.dict(sys.modules, {"certifi": None}):
            context = network.create_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_invalid_explicit_ca_bundle_fails_instead_of_falling_back(self):
        context = Mock()
        context.load_verify_locations.side_effect = FileNotFoundError("Missing CA file")
        with patch.dict(os.environ, {"SSL_CERT_FILE": "missing.pem"}), \
                patch.object(network.ssl, "create_default_context", return_value=context):
            with self.assertRaises(FileNotFoundError):
                network.create_ssl_context()

    def test_urlopen_passes_verified_context(self):
        context, response = Mock(), Mock()
        with patch.object(network, "create_ssl_context", return_value=context), \
                patch.object(network.urllib.request, "urlopen", return_value=response) as opener:
            self.assertIs(network.urlopen("https://example.invalid", timeout=7), response)
        opener.assert_called_once_with("https://example.invalid", timeout=7, context=context)

    def test_certificate_failure_explains_repair_and_never_retries_insecurely(self):
        failure = urllib.error.URLError(ssl.SSLCertVerificationError(1, "unable to get local issuer certificate"))
        with patch.object(network, "create_ssl_context", return_value=Mock()), \
                patch.object(network.urllib.request, "urlopen", side_effect=failure) as opener:
            with self.assertRaisesRegex(urllib.error.URLError, "SSL_CERT_FILE"):
                network.urlopen("https://example.invalid")
        self.assertEqual(opener.call_count, 1)

    def test_other_network_errors_are_preserved(self):
        failure = urllib.error.URLError("DNS unavailable")
        with patch.object(network, "create_ssl_context", return_value=Mock()), \
                patch.object(network.urllib.request, "urlopen", side_effect=failure):
            with self.assertRaises(urllib.error.URLError) as caught:
                network.urlopen("https://example.invalid")
        self.assertIs(caught.exception, failure)


class ZipDownloadCertificateTests(unittest.TestCase):
    def setUp(self):
        try:
            from api import datasets
        except ImportError:
            self.skipTest("Install the api extra")
        self.datasets = datasets

    def download(self, connection, context):
        settings = types.SimpleNamespace(download_timeout=10, max_upload_bytes=1024)
        addresses = [(2, 1, 6, "", ("1.1.1.1", 443))]
        with patch.object(self.datasets.socket, "getaddrinfo", return_value=addresses), \
                patch.object(self.datasets, "create_ssl_context", return_value=context), \
                patch.object(self.datasets.http.client, "HTTPSConnection", return_value=connection) as factory:
            result = self.datasets.download_zip("https://example.invalid/dataset.zip", settings)
        self.assertIs(factory.call_args.kwargs["context"], context)
        self.assertEqual(factory.call_args.args[0], "example.invalid")
        return result

    def test_zip_uses_verified_context_and_original_tls_hostname(self):
        connection, context, response = Mock(), Mock(), Mock()
        response.status = 200
        response.getheader.return_value = "3"
        response.read.side_effect = [b"zip", b""]
        connection.getresponse.return_value = response
        result = self.download(connection, context)
        self.assertIsInstance(result, BytesIO)
        self.assertEqual(result.read(), b"zip")
        connection.close.assert_called_once_with()

    def test_zip_certificate_failure_is_actionable_and_connection_closed(self):
        connection = Mock()
        connection.request.side_effect = ssl.SSLCertVerificationError(1, "unable to get local issuer certificate")
        with self.assertRaisesRegex(ValueError, "SSL_CERT_FILE"):
            self.download(connection, Mock())
        connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

import ssl

from autointerp.tls import RELAX_ENV, ssl_context


def test_default_keeps_strict_flag(monkeypatch):
    monkeypatch.delenv(RELAX_ENV, raising=False)
    ctx = ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname
    if hasattr(ssl, "VERIFY_X509_STRICT") and ssl.create_default_context().verify_flags & (
        ssl.VERIFY_X509_STRICT
    ):
        assert ctx.verify_flags & ssl.VERIFY_X509_STRICT


def test_env_relaxes_only_strict_flag(monkeypatch):
    monkeypatch.setenv(RELAX_ENV, "1")
    ctx = ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname
    assert not ctx.verify_flags & ssl.VERIFY_X509_STRICT

import ssl

from autointerp.tls import ssl_context


def test_only_strict_flag_is_relaxed():
    ctx = ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname
    assert not ctx.verify_flags & ssl.VERIFY_X509_STRICT

"""TLS context for a machine behind a TLS-inspecting proxy.

The proxy CA lacks a *critical* Basic Constraints extension, which Python's
VERIFY_X509_STRICT flag rejects. Only that flag is cleared (decision of
2026-11-19). Chain, signature, hostname and expiry are still verified.
Scope: the project's own HTTP clients only.
"""

import os
import ssl


def ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    return ctx

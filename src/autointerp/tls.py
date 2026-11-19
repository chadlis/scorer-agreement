"""TLS context for the project's HTTP clients.

Behind a TLS-inspecting proxy whose CA lacks a *critical* Basic Constraints
extension, Python's VERIFY_X509_STRICT flag rejects every connection. Setting
AUTOINTERP_RELAX_X509_STRICT=1 clears that single flag. Chain, signature, hostname
and expiry are still verified. Without the variable, the default context is used.
"""

import os
import ssl

RELAX_ENV = "AUTOINTERP_RELAX_X509_STRICT"


def ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)
    if os.environ.get(RELAX_ENV) == "1":
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    return ctx

import ssl


# Workaround for SSL certificate issues on some systems
# "certificate verify failed: unable to get local issuer certificate"
# Use certifi's CA bundle as a safe fallback when available.
try:
    import certifi
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    # urllib and other stdlib modules use ssl._create_default_https_context to
    # build HTTPS contexts. Overriding it ensures downloads validate using
    # certifi's CA bundle when available.
    ssl._create_default_https_context = lambda: _ssl_ctx
except Exception:
    # If certifi isn't installed or something fails, leave the default
    # context unchanged. The recommended fix then is to install certifi or
    # run the system-specific certificate installer (e.g. the
    # "Install Certificates.command" that ships with some Python installers
    # on macOS).
    pass

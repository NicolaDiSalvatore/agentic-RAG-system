"""Process-wide setup that must happen before any network library is imported."""

import truststore

# Enterprise proxies use root certificates that the operating system trusts but
# certifi does not, so every entry point verifies against the system store.
truststore.inject_into_ssl()

# Imported for its side effect: it applies the offline setting before Hugging
# Face libraries load, because they read it once at import time.
from src import config  # noqa: E402, F401

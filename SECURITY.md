# Security policy

## Supported versions

Security fixes are provided for the latest release and the current `main` branch.

## Reporting a vulnerability

Do not open a public issue for path traversal, arbitrary file access, unsafe
deserialization, command execution, browser/API authorization bypass, or another
suspected vulnerability. Use GitHub's **Security → Report a vulnerability** form
for this repository and include reproduction steps, impact, affected versions,
and any proposed mitigation.

Please avoid accessing data that is not yours and do not include model weights,
personal media, tokens, or other secrets in a report. Maintainers will acknowledge
a report when received and coordinate disclosure after a fix is available.

This project exposes checkpoint-management routes to the ComfyUI server. ComfyUI
should not be exposed directly to an untrusted network without authentication and
network-layer access controls.

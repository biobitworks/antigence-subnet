# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 10.x    | Yes       |
| < 10.0  | No        |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

To report a vulnerability, email **security@biobitworks.com** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

We will acknowledge receipt within 72 hours and provide an initial assessment within 7 days. Critical vulnerabilities affecting miner or validator key material will be prioritized.

## Bittensor Ecosystem: Typosquatting Warning

This subnet's official package is `antigence-subnet` published at [github.com/biobitworks/antigence-subnet](https://github.com/biobitworks/antigence-subnet).

**Be cautious of:**

- PyPI packages with similar names (e.g., `antigence`, `antigence_subnet`, `antigence-bittensor`) — this project is NOT published on PyPI
- Forked repos that modify wallet handling, key generation, or weight-setting logic
- Modified `neurons/` scripts that redirect earnings or exfiltrate keys
- Docker images not built from this repository's official Dockerfile

**Before running any Bittensor subnet code:**

1. Verify you cloned from `github.com/biobitworks/antigence-subnet`
2. Check the commit signature or tag against this repo
3. Review any modifications to `neurons/validator.py` and `neurons/miner.py`
4. Never paste your coldkey mnemonic into any script or website

## Dependency Security

This project pins `bittensor==10.2.0`. Known dependency advisories:

- `setuptools~=70.0` is pinned by the bittensor SDK and has a known path traversal CVE (PYSEC-2025-49). The vulnerable API (`PackageIndex`/`easy_install`) is deprecated and not used at runtime. This is an upstream constraint.

Run `pip-audit` against your installed environment to check for current advisories.

## Public Data Notes

Some evaluation fixtures in `data/` contain synthetic credential-like strings (fake API keys, tokens, passwords). These are adversarial test cases for anomaly detection, not real secrets.

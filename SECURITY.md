# Security Policy

This document outlines the security policy for the Awesome LLM Tools repository
and explains how to report security concerns responsibly.

---

## Scope

Awesome LLM Tools is a curated markdown list, not a software package or
executable application. As such, the security considerations are different from
a typical software project.

Security concerns relevant to this repository include:

- Malicious or deceptive URLs in tool listings
- Tools that have been compromised or taken over by bad actors since listing
- Scripts in this repository (`scripts/`) that contain unsafe code
- GitHub Actions workflows (`.github/workflows/`) that could be exploited
- Sensitive information accidentally committed to the repository

Security concerns outside the scope of this policy:

- Vulnerabilities in the tools listed in README.md (report those directly
  to the respective tool maintainers)
- General feedback about tool quality or accuracy (use GitHub Issues instead)

---

## Reporting a Vulnerability

Do not open a public GitHub issue for security vulnerabilities. Public
disclosure before a fix is in place puts the community at risk.

To report a security concern, use one of the following methods:

### GitHub Private Vulnerability Reporting (preferred)

GitHub's private vulnerability reporting allows you to submit a security
report directly to the maintainers without public disclosure.

1. Go to the repository Security tab
2. Click "Report a vulnerability"
3. Fill in the details and submit

This keeps the report confidential until a fix is ready.

### Email

If you are unable to use GitHub's private reporting, send a detailed
report to the maintainer directly. Use the contact information listed
on the maintainer's GitHub profile.

---

## What to Include in a Report

A good security report helps us triage and resolve the issue faster.
Please include:

- A clear description of the vulnerability or concern
- The location of the issue (file name, line number, URL, or section)
- Steps to reproduce or verify the issue
- The potential impact if left unaddressed
- Any suggested fix or mitigation, if you have one

---

## Response Timeline

We aim to respond to all security reports according to the following schedule:

| Stage                       | Target Timeframe |
| --------------------------- | ---------------- |
| Acknowledge receipt         | Within 48 hours  |
| Initial assessment          | Within 5 days    |
| Resolution or status update | Within 14 days   |

Complex issues may take longer to resolve. We will keep you informed of
progress throughout the process.

---

## Disclosure Policy

We follow coordinated disclosure practices:

- Security reports are kept confidential until a fix is released
- We will notify you when the issue has been resolved
- We will credit reporters in the fix commit or release notes unless
  you prefer to remain anonymous
- We ask that reporters allow reasonable time for a fix before any
  public disclosure

---

## Supported Versions

As a curated list rather than a versioned software package, all changes
apply to the current state of the `main` branch. There are no legacy
versions to maintain.

---

## Safe Harbor

We consider security research conducted in good faith to be authorized
and will not pursue legal action against researchers who:

- Report vulnerabilities through the channels described above
- Avoid accessing, modifying, or deleting data that does not belong to them
- Do not disrupt the repository or its associated services
- Act in accordance with this policy throughout the process

---

## Attribution

Reporters who responsibly disclose security issues will be acknowledged
in the relevant fix commit or changelog entry, unless they request
anonymity.

---

_Last updated: April 2026_

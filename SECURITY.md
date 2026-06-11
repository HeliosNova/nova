# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.6.x   | Yes       |
| < 1.6   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Nova, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. Email the maintainer directly with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

You should receive an acknowledgment within 48 hours. We aim to release patches for critical vulnerabilities within 7 days.

## Security Architecture

Nova implements multiple layers of security:

### Access Tiers (`SYSTEM_ACCESS_LEVEL`)
- **sandboxed** (default): Most restrictive. Shell blocks system + interpreter commands. File ops limited to `/data`. Code execution blocks os/subprocess/socket.
- **standard**: Blocks system commands. File access expanded to `/data`, `/tmp`, `/home/nova`.
- **full**: Only container-escape commands blocked.
- **none**: All restrictions disabled.

### Prompt Injection Detection
Two layers:
- **User queries** — mandatory, fail-closed detection in the chat pipeline. Not configurable: if the detector errors, the query is blocked rather than passed through.
- **External content** (web search, HTTP fetch, MCP tools, browser, knowledge base) — heuristic scanning plus always-on sanitization of all tool outputs before they re-enter the prompt. The external-content scan is controlled by `ENABLE_INJECTION_DETECTION`.

### Authentication
- Bearer token auth on all API endpoints (`NOVA_API_KEY`)
- Per-IP rate limiting with lockout on repeated auth failures
- Channel-specific user allowlisting (Discord, Telegram, WhatsApp, Signal)
- No cookies or sessions — auth is a request header, so CSRF does not apply (a cross-origin page cannot attach the bearer token, and CORS restricts allowed origins)

### Skill Signing
Skills can be signed with HMAC-SHA256. Set `REQUIRE_SIGNED_SKILLS=true` (default) to reject unsigned skill imports.

### Docker Hardening
- Non-root user (`nova`, uid 1000)
- Read-only root filesystem
- `no-new-privileges` security option
- All capabilities dropped
- Resource limits enforced

## Best Practices

1. Always set `NOVA_API_KEY` in production
2. Keep `SYSTEM_ACCESS_LEVEL=sandboxed` unless you have a specific need
3. Enable `ENABLE_INJECTION_DETECTION=true`
4. Set channel allowlists for all enabled messaging channels
5. Never commit `.env` files — use `.env.example` as a template
6. Rotate API keys and tokens regularly

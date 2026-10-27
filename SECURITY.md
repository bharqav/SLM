# Security Policy

## Supported Versions

This project is actively maintained. Security patches are applied to the latest release on the `main` branch.

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. **Do NOT** post details of the vulnerability in public forums or social media
3. **DO** email the maintainers directly at security@example.com (replace with actual contact)

Include the following information:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Security Best Practices

### Model Artifacts

- **Never commit** real model weights or sensitive training data to version control
- Use `.gitignore` to exclude `.pt`, `.bin`, `.safetensors` files
- Store large artifacts in dedicated storage (S3, GCS, HuggingFace Hub)
- Verify checksums of downloaded artifacts before loading

### API Server

- Always set a strong `API_KEY` in production
- Never expose the API server directly to the internet without authentication
- Use HTTPS in production (reverse proxy with nginx/Caddy)
- Rate-limit requests to prevent abuse

### Dependencies

- Keep dependencies up to date: `pip install -U -r requirements.txt -c constraints.txt`
- Run security scans: `make security` or `python -m pip_audit`
- Review dependency updates before merging

### Data Privacy

- Training data may contain sensitive information
- Follow data licensing terms (CC BY-NC-4.0, CC BY-NC-SA)
- Do not use the model for processing personal data without appropriate safeguards

## Known Limitations

- The Qwen merged weights in `slm-ft/merged_fp16/` are placeholder stubs, not real weights
- The empathetic model checkpoint is a placeholder JSON file
- No real training corpus is included in this repository

These limitations do not constitute security vulnerabilities but should be understood by users of the repository.

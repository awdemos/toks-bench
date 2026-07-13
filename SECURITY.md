# Security Policy

## Scope

`toks-bench` is a **local-only** benchmark harness for OpenAI-compatible LLM
inference servers. It is designed to run on a trusted operator workstation and
to communicate with inference servers on `localhost` or `127.0.0.1`.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Model

- Treat `config.yaml`, all prompt files, sweep scripts, and result files as
  **trusted input**. An attacker who can write these files can influence which
  local servers are benchmarked and which files are read.
- By default `toks-bench` only allows provider `base_url` values pointing to
  `localhost`, `127.0.0.1`, or `::1`. Use `--allow-internal-urls` only when you
  genuinely need to benchmark servers on a private LAN, and never point
  providers at cloud metadata endpoints such as `169.254.169.254`.
- Prompt files referenced by `file:` in `config.yaml` must reside in the same
  directory as `config.yaml` (or a subdirectory). Path traversal via `..` or
  symlinks escaping the config directory is rejected.
- Result output paths (`--output`, `--csv`, `--report`) must remain within the
  current working directory. Path traversal is rejected.
- Shell sweep scripts run inference servers in Docker containers. Container
  images should be pinned to a digest (`image@sha256:...`) via the
  `VLLM_IMAGE` / `TRITON_IMAGE` environment variables to avoid mutable-tag
  supply-chain attacks.
- The Nemotron-3-Nano reasoning-parser plugin is downloaded from HuggingFace
  and verified against a pinned SHA-256 hash before being mounted into the
  container.

## Reporting a Vulnerability

If you discover a security issue in `toks-bench`, please open a private
security advisory on GitHub or email the maintainers directly. Do not file a
public issue for undisclosed vulnerabilities.

We aim to acknowledge reports within 5 business days and to release a fix or
mitigation within 30 days for critical issues.

## Security Hardening Checklist

- [ ] Install dependencies with a hash-verified lockfile:
      `uv sync --locked` or `pip install -r requirements.txt --require-hashes`.
- [ ] Pin container images to a digest in sweep scripts.
- [ ] Set `TENSORRT_LLM_VERSION` to a pinned release in
      `scripts/install-trtllm.sh`.
- [ ] Bind native inference servers to `127.0.0.1` unless remote access is
      required.
- [ ] Run CI security gates (`bandit`, `ruff --select S`, `pip-audit`) before
      merging changes.

# Privacy and local configuration

Use environment variables or an untracked local configuration file for credentials.
Do not commit real chat transcripts, request bodies, authorization headers, cache
snapshots, or diagnostic archives containing user content. Use synthetic prompts
for reproducible examples and tests.

Before sharing diagnostics, remove access tokens, personal paths, user identifiers,
private endpoints, and prompt/output text. Prefix caches and state images can encode
conversation content and should be treated as private data.

The serving API supports `--api-key`; configure authentication and an appropriate
network boundary before allowing other machines to connect. See
[serving authentication](docs/serving.md#authentication-and-cors).

Before publishing changes, review staged files and scan both the current files and
Git history with a secret scanner. Ignore rules do not remove files already tracked
by Git. If a real credential has been committed, revoke or rotate it and remove it
from the affected history before sharing the repository.

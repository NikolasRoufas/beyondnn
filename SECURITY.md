# Security Policy

## Supported versions

| version | supported |
|---|---|
| 0.1.x | yes (fixes land on `main` and in the next 0.1.x release) |
| earlier development snapshots | no |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub:
1. Go to **Security → Report a vulnerability** in this repository (GitHub private vulnerability reporting).
2. Describe the issue, the affected version or commit, and how to reproduce it.

Reports are visible only to the maintainers. You should receive an acknowledgement within a week.

## Security-relevant design points

- **Loading traces.** Tensor sidecar files are loaded with `torch.load(weights_only=True)`, so loading a trace never executes pickled code. Saved traces and reports are plain JSON plus that tensor file. Do not load traces with `weights_only=False`.
- **Integrity is not authenticity.** Record ids are content hashes, and loading re-validates every record, so accidental corruption and inconsistent edits are refused. A hash is not a signature: anyone who can write a trace can write a self-consistent one. Treat evidence from untrusted sources accordingly.
- **Untrusted models.** Tracing and interventions execute the model's `forward`. BeyondNN does not sandbox model code.
- **Generated text.** Any model- or template-generated label is untrusted content and is always labelled `GENERATED`.

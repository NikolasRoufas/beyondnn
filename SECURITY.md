# Security Policy

## Supported versions

BeyondNN has not been released. There are no supported versions yet.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Report them privately to
**[INSERT SECURITY CONTACT]**, or via GitHub private vulnerability reporting once the repository is public.

## Known security-relevant design points

- **Loading traces.** Tensor sidecar files are loaded with `torch.load(weights_only=True)`, so
  loading a trace never executes pickled code. Do not load traces with `weights_only=False`.
- **Untrusted models.** Tracing executes the model's `forward`. BeyondNN does not sandbox model code.
- **Generated text.** Optional LLM-generated summaries (a future feature) are untrusted content and are
  always labelled `GENERATED`.

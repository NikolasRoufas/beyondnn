"""Prose for an :class:`~beyondnn.audits.report.AuditReport`, rendered only from its
structured data. The wording states what the recorded evidence establishes under the
declared plan; it never calls an explanation trustworthy, reliable, or correct."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .report import AuditReport, ClaimAudit, ConceptAudit, Finding

_WIDTH = 100


def _finding(f: Finding, indent: str = "    ") -> list[str]:
    head = f"{indent}[{f.severity.value.upper()}] {f.kind.value.upper()} {f.code}"
    if f.axis:
        head += f" (axis: {f.axis})"
    lines = [head, f"{indent}  {f.detail}"]
    if f.samples:
        shown = ", ".join(f.samples[:6]) + (
            f", ... ({len(f.samples)} samples)" if len(f.samples) > 6 else ""
        )
        lines.append(f"{indent}  samples: {shown}")
    if f.records:
        shown = ", ".join(f.records[:4]) + (
            f", ... ({len(f.records)} records)" if len(f.records) > 4 else ""
        )
        lines.append(f"{indent}  records: {shown}")
    return lines


def _claim(c: ClaimAudit) -> list[str]:
    decl = c.claim
    about = (
        f"selection {decl.selection.method} at {decl.selection.site.module or 'input'}"
        + (f" (k={decl.selection.k})" if decl.selection.k is not None else " (any k)")
        if decl.selection is not None
        else f"subject {decl.subject.site.module or 'input'}"  # type: ignore[union-attr]
    )
    lines = [
        f"  {decl.name}: {decl.relation.value.upper()} {decl.target.metric} "
        f"({decl.scope.value}; {about}; requirement {decl.requirement})",
        f"    statement (declared, not evidence): {decl.statement}",
    ]
    if decl.invariant_over:
        lines.append(
            "    asserted invariant over: "
            + ", ".join(f"{i.axis.value} (>= {i.min_values} values)" for i in decl.invariant_over)
        )
    if c.per_sample:
        total = sum(n for _, n in c.distribution)
        dist = ", ".join(f"{s.upper()} {n}" for s, n in c.distribution)
        lines.append(f"    per-sample standings ({total} declared samples): {dist}")
        lines.append("    (a distribution over samples; no claim-level truth value is derived)")
        if c.counterexamples:
            lines.append(
                f"    counterexample samples: {', '.join(c.counterexamples[:10])}"
                + (" ..." if len(c.counterexamples) > 10 else "")
            )
    else:
        assert c.standing is not None
        g = c.groups[0]
        lines.append(f"    STANDING: {c.standing.value.upper()} (verdict {g.verdict})")
        for t in g.tests:
            alt = f" [re-evaluated: {t.alternative}]" if t.alternative else ""
            lines.append(f"      {t.protocol}: {t.outcome.upper()} ({t.result}){alt}")
    for f in c.findings:
        lines += _finding(f)
    if c.limitations:
        lines.append("    limitations: " + ", ".join(f"{code} x{n}" for code, n in c.limitations))
    return lines


def _concept(k: ConceptAudit) -> list[str]:
    lines = [
        f"  concept {k.label!r} ({k.label_source} label; {k.concept.concept}): asserted "
        f"{k.concept.asserted.value.upper()} under {k.concept.policy.name}/v"
        f"{k.concept.policy.version}",
        f"    STANDING: {k.standing.value.upper()}",
    ]
    for vid, status, unmet in k.validations:
        lines.append(f"      validation {vid}: {status.upper()}")
        for reason in unmet:
            lines.append(f"        unmet: {reason}")
    for t in k.tests:
        axes = dict(t.axes)
        lines.append(
            f"      {t.protocol}: {t.outcome.upper()} (null {axes.get('null')}; "
            f"replacement {axes.get('replacement')})"
        )
    if k.false_positives or k.false_negatives:
        lines.append(
            f"    counterexamples kept: {len(k.false_positives)} false positive(s), "
            f"{len(k.false_negatives)} false negative(s)"
        )
    for f in k.findings:
        lines += _finding(f)
    return lines


def render(report: AuditReport) -> str:
    plan = report.plan
    cov = report.coverage
    ev = report.evidence
    lines = [
        f"AUDIT {plan.name} (plan {plan.id})",
        "What this is: a classification of declared claims by the recorded, re-derived,",
        "in-scope evidence under the declared plan. It is not a score and not a verdict on",
        "whether any explanation should be believed.",
        "",
        "SCOPE",
        f"  checkpoint {plan.checkpoint}",
        "  declared model: "
        + ("none" if plan.declared_model is None else repr(plan.declared_model)),
        f"  {len(plan.samples)} declared sample(s); {len(plan.datasets)} concept dataset(s)",
        f"  evidence: {ev.traces} trace(s), {ev.records} record(s), {ev.results} claim-test "
        f"result(s) ({ev.results_included} included, {ev.results_excluded} excluded); digest "
        f"{ev.digest[:23]}...",
        "",
        "CLAIMS",
    ]
    for c in report.claims:
        lines += _claim(c)
    if report.concepts:
        lines += ["", "CONCEPTS"]
        for k in report.concepts:
            lines += _concept(k)
    if report.findings:
        lines += ["", "EVIDENCE PROBLEMS (excluded; never merged)"]
        for f in report.findings:
            lines += _finding(f, "  ")
    if report.diagnostics:
        lines += ["", "DIAGNOSTICS (context; they decide nothing)"]
        for d in report.diagnostics:
            status = "re-derived" if d.verified else "NOT re-derivable"
            outcomes = ", ".join(f"{a}: {o}" for a, o in d.outcomes) or "no aspects"
            lines.append(f"  {d.protocol} over {d.samples} sample(s) ({status}): {outcomes}")
    lines += [
        "",
        "COVERAGE (coverage, not confidence)",
        "  claims by standing: "
        + (
            ", ".join(f"{s.upper()} {n}" for s, n in cov.claims_by_standing) or "none single-valued"
        ),
    ]
    for name, n, evaluated in cov.per_sample_claims:
        lines.append(f"  {name}: evidence on {evaluated} of {n} declared samples")
    lines += [
        f"  concepts evaluated: {cov.concepts_evaluated} of {cov.concepts}",
        f"  evidence statuses present: {', '.join(cov.evidence_statuses) or 'none'}",
        f"  protocols run: {', '.join(cov.protocols_run) or 'none'}",
        f"  required protocols never run: {', '.join(cov.protocols_required_not_run) or 'none'}",
        "  declared invariances not tested: "
        + (", ".join(f"{c} ({a})" for c, a in cov.untested_invariances) or "none"),
        "  counterexample caps: " + ", ".join(f"{k} {v}" for k, v in cov.counterexample_caps),
    ]
    if report.limitations:
        lines += ["", "LIMITATIONS (codes in the evidence)"]
        lines.append("  " + ", ".join(f"{code} x{n}" for code, n in report.limitations))
    return "\n".join(lines)

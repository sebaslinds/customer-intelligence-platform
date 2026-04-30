# Decision Engine Documentation

## Purpose

The Decision Engine converts business metrics and operational signals into prioritized alerts, recommendations, and executive explanations.

It is designed to answer:

```text
What should the business do next, based on the current signals?
```

## Pipeline

```text
Business metrics
    +
Operational checks
    +
Detected anomalies
        |
        v
Decision rules
        |
        v
Prioritized decisions
        |
        v
Gemini or local explanation
```

## What Counts As An Anomaly?

An anomaly is a signal that falls outside an expected healthy range. In this project, anomalies are not random numbers selected by the user. They are generated from business and operational checks.

Examples:

- reorder rate drops below the healthy threshold
- churn rate moves into a high-risk range
- days between orders become unusually long
- data quality checks fail
- API health check fails
- revenue proxy drops sharply

## Input Contract

```json
{
  "data": {
    "reorder_rate": 0.22,
    "churn_rate": 0.61,
    "days_between_orders": 24,
    "data_quality_failures": 0,
    "api_health": "ok",
    "revenue_proxy_delta": -0.12
  },
  "anomalies": [],
  "use_gemini": true,
  "language": "en"
}
```

## Rule Summary

| Signal | Rule | Priority | Decision Type |
| --- | --- | --- | --- |
| `reorder_rate` | Below 0.35 | High | Recommendation |
| `churn_rate` | 0.50 to 0.69 | High | Recommendation |
| `churn_rate` | 0.70 or higher | Critical | Alert |
| `days_between_orders` | 21 days or higher | High | Recommendation |
| `data_quality_failures` | Greater than 0 | Critical | Alert |
| `api_health` | Not healthy | Critical | Alert |
| `revenue_proxy_delta` | Negative movement | Medium or High | Recommendation |

## Priority Levels

| Priority | Meaning |
| --- | --- |
| Low | Monitor, no urgent action |
| Medium | Review and plan an action |
| High | Act soon to reduce business risk |
| Critical | Immediate operational or business risk |

## Explanation Sources

The Decision Engine can explain its output in two ways:

| Source | When Used |
| --- | --- |
| Gemini | `GEMINI_API_KEY` is configured and `use_gemini` is true |
| Local fallback | Gemini is unavailable, disabled, or returns an error |

The local fallback is deterministic and rule-based. This keeps the platform reliable even when the LLM provider is unavailable.

## Example Response

```json
{
  "decisions": [
    {
      "decision_type": "recommendation",
      "priority": "high",
      "title": "Launch retention intervention",
      "action": "Target high-risk customers with reorder reminders.",
      "rationale": "Low reorder rate and long purchase gaps indicate declining engagement."
    }
  ],
  "explanation_source": "gemini",
  "explanation": "The highest-priority decision is to launch a retention intervention because reorder engagement is weak and purchase gaps are widening."
}
```

## Dashboard Interpretation

The dashboard shows:

- detected anomalies
- priority alerts
- decision recommendations
- explanation source
- rule evidence
- carbon estimate for the AI explanation path

The business scenario sliders are used to simulate customer behavior inputs. Operational signals such as API health and data quality are detected from live checks and are not manually selected.

## Known Limitations

- Rule thresholds are currently static.
- The decision engine is advisory and does not execute automated actions.
- Gemini explanations summarize the rule output but do not change the underlying decision rules.
- Real production systems should add audit logs, ownership, notification channels, and approval workflows.

## Recommended Next Improvements

- Add Slack or email alert routing.
- Add action history and acknowledgement status.
- Persist decisions to Snowflake for auditability.
- Add rule configuration from YAML or database tables.
- Add severity scoring based on trend direction and business impact.
- Add automated daily decision reports.

# CaptainFix QA Project — Claude Code Rules

## 1. Never read error screenshot files when debugging

Do **not** open `error_*.png` files to debug test failures. Fix failures by reading the error traceback and the test code (`tests/TC*.py`) directly.

## 2. Always propagate fixes from generated tests back into the prompt templates

The tests (`tests/TC*.py`) and the plan (`output/plan.json`) are LLM-generated. A fix applied only to the output file is lost on the next regeneration run.

- Fixing how a Selenium step is implemented → also edit `executor.py` (`SYSTEM_PROMPT_TEMPLATE`).
- Changing which steps are planned or their structure → also edit `planner.py`.
- Keep `plan.json` and the matching `TC*.py` consistent when a change touches steps or expected values.

After editing the template, verify that `SYSTEM_PROMPT_TEMPLATE.format(...)` still works — escape any literal braces as `{{}}`.

## 3. Never generate positive submission tests for CAPTCHA-gated forms

The site uses ALTCHA (`<altcha-widget>`), a proof-of-work captcha injected by JS into the DOM (possibly shadow DOM). A test that tries to submit a CAPTCHA-gated form will always fail — do not add such cases to the plan.

## 4. Pick visible, non-zero-size elements — never rely on bare index or first match

When locating elements that may have hidden duplicates (footer links, icons, toggles), always filter for visible, non-zero-size matches:

- Footer links: iterate `reversed(find_elements(...))`, pick the last **visible** non-zero-size match.
- Multi-match click targets (e.g. `.show-searchbox`): pick the **first** visible non-zero-size match.
- Heading checks: do not require a specific `h2` — verify a visible content element or footer instead.

## 5. Each Navigation test must use a distinct `nav_pattern`

No two Navigation test cases in the same plan may share the same `nav_pattern` value (A/B/C/D). "Same interaction shape" counts as a duplicate even if the final assertion differs.

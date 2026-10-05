# Licence decision: owner action required

No `LICENSE` file has been added. This was deliberate: the software implements the subject of a pending patent application (IN 202541122226, applicant VIT), and choosing a licence affects both the patent strategy and VIT's rights.

Options to decide with counsel and VIT:

1. **Proprietary / all rights reserved** (typical for a commercial SaaS). Make the GitHub repositories private. They are public today, and they contain the legacy prototypes, including the historical commits with the rotated keys.
2. **Source-available** (for example BSL or a custom licence) if public visibility matters for credibility, with commercial use reserved.
3. **Open source:** permissive licences include patent grants (Apache-2.0 has an explicit one) that may conflict with patent monetisation. Not recommended without advice.

Also decide:
- whether to purge the leaked keys and personal recordings from git history (needs a force-push);
- whether to archive `futuristic-ai-interviewer` now that its history is imported here.

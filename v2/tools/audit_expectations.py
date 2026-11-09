"""Expected audit findings and clean passages -- the audit half of the gold set.

Retrieval metrics answer "did it find the right clauses". These answer the
question the product actually exists for: "did it find the right VIOLATIONS,
and only those".

How these were authored
-----------------------
From the regulation, not from any model's output. Each expectation names the
obligation the passage fails and quotes the text that fails it, so a reader who
disagrees can check the reasoning rather than take it on trust. Nothing here
was produced by, or validated against, the system under test.

What these can and cannot measure
---------------------------------
EXPECTED is not exhaustive. A real CER review would surface more than four
findings; these are the ones defensible enough to hold a system to. So RECALL
here means "of the violations we are certain are present, how many were found"
-- a lower bound on the true recall, not an estimate of it.

CLEAN means "a reviewer would not expect a finding here", not "provably
compliant". These are identification, administrative and descriptive sections
where the information the regulation asks for is present and no obligation is
plausibly breached. A finding raised on one of them is very likely a false
positive, which is what makes them the most useful part of this set -- most
audit evaluations measure only recall and never notice a system that flags
everything.

MATCHING
--------
A predicted finding matches an expectation when it is on the same passage AND
cites a clause inside the expectation's scope. The quoted text is recorded for
human review but is NOT required to match: two reviewers can cite the same
breach from different sentences of the same passage.
"""

from __future__ import annotations

# --- violations that are genuinely present --------------------------------

EXPECTED_FINDINGS: dict[str, list[dict]] = {
    "CER.4.3.2.1": [
        {
            "clause_scope": "Art.61.4",
            "summary": "Declares the device implantable, then states no clinical "
                       "investigation was performed, citing Article 61 generically.",
            "reasoning": "Article 61(4) requires clinical investigations for "
                         "implantable and class III devices. The exemption in 61(6) "
                         "is narrow and 61(7) requires the reliance on it to be "
                         "justified in the clinical evaluation report. The passage "
                         "asserts the device IS implantable and then declines the "
                         "investigation without invoking either provision.",
            "evidence": "this technology has indeed been in routine clinical use for "
                        "over a decade. Hence, according to MDR article 61, no "
                        "clinical investigation was performed",
            "min_severity": "High",
        }
    ],
    "CER.2.11": [
        {
            "clause_scope": "Annex.I.23",
            "summary": "Blanket claim that no absolute contraindications exist, with "
                       "no supporting analysis.",
            "reasoning": "Annex I s23 requires contraindications to be stated in the "
                         "information supplied with the device. A one-sentence "
                         "negative claim for an indwelling ureteral stent -- a device "
                         "class with well-documented contraindications such as "
                         "untreated urinary infection -- is an unsupported assertion "
                         "rather than a disclosure.",
            "evidence": "There are no known absolute contraindications for this device.",
            "min_severity": "Medium",
        }
    ],
    "CER.2.19": [
        {
            "clause_scope": "Annex.II.1",
            "summary": "Names the Long Term device in a clinical evaluation of the "
                       "Short Term device.",
            "reasoning": "Annex II s1 requires an accurate device description. The "
                         "section identifies 'Double J Stent/Kit (Long Term)' while "
                         "the entire report concerns the Short Term variant. Either "
                         "the wrong device is described or the scope is wrong; both "
                         "are documentation defects a notified body would raise.",
            "evidence": "The Double J Stent/Kit (Long Term) is a first-generation device",
            "min_severity": "Medium",
        }
    ],
    "CER.4.3.2.2#1": [
        {
            "clause_scope": "Art.83",
            "summary": "Post-market surveillance rests on adverse-event data from "
                       "generic devices rather than the device itself.",
            "reasoning": "Article 83(1) requires a post-market surveillance system "
                         "FOR EACH DEVICE, proportionate to its risk class. Generic "
                         "device data may supplement it but cannot substitute for "
                         "surveillance of the subject device.",
            "evidence": "Adverse event data of generic devices",
            "min_severity": "High",
        }
    ],
}


# --- passages where a finding would be a false positive -------------------

CLEAN_PASSAGES: list[str] = [
    "CER.2.1",      # device identification
    "CER.2.2",      # manufacturer name and address
    "CER.2.3",      # administrative particulars
    "CER.2.4",      # device description
    "CER.2.5",      # classification, correctly reasoned from Annex VIII
    "CER.2.6",      # technical specifications
    "CER.2.7#1",    # components and materials
    "CER.2.7#2",
    "CER.2.8",      # mechanism of action
    "CER.2.9",      # intended use
    "CER.2.10",     # indications
    "CER.2.12",     # intended patient population
    "CER.2.13",     # intended users, stated with required competence
    "CER.3.3",      # clinical background
    "CER.3.6",      # alternative treatment options -- satisfies Art. 61(3)(c)
]


def expectation_count() -> int:
    return sum(len(v) for v in EXPECTED_FINDINGS.values())

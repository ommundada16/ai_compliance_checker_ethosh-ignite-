"""Expert mapping: CER section -> governing EU MDR 2017/745 clauses.

This is the human-authored half of the gold set, and the part that carries the
most judgement. It is deliberately a hand-written rubric rather than anything
derived from a retrieval system, because the whole point of the retrieval
metrics is to test a retriever against ground truth. Ground truth produced BY a
retriever would only measure that retriever's self-consistency.

The mapping follows the standard structure of a Clinical Evaluation Report
under MDR Article 61 and Annex XIV Part A, as laid out in MEDDEV 2.7/1 rev 4
(still the working template the industry uses under MDR). Each CER section has
a well-understood regulatory home:

    device description / intended purpose  -> Annex II s1, Annex I s1
    labelling, IFU, contraindications      -> Annex I s23
    risk and side-effects                  -> Annex I s3, s4, s8
    equivalence                            -> Article 61(5), Annex XIV Part A s3
    clinical evaluation method             -> Article 61(1)(3), Annex XIV Part A s1
    post-market surveillance / PMCF        -> Articles 83-86, Annex III, Annex XIV Part B

GRADED RELEVANCE
----------------
Two grades, because nDCG needs them and because "relevant" is genuinely not
binary here:

    primary   (grade 2)  the clause a regulator would cite first when assessing
                         this section. Missing it is a real retrieval failure.
    secondary (grade 1)  a clause that legitimately bears on the section and
                         that a reviewer would accept as useful context, but
                         whose absence is not a failure.

Recall@k is computed over PRIMARY only. nDCG uses both grades. Keeping the two
separate stops a retriever from scoring well by dredging up loosely-related
context while missing the clause that actually governs.

PROVENANCE
----------
Labels derived from this file are tagged "expert_map". A separate pass
(tools/propose_labels_llm.py) asks an independent strong model for candidates;
where the two disagree, the disagreement is recorded rather than silently
resolved. See eval_data/README.md for how conflicts are settled.
"""

from __future__ import annotations

# fmt: off
SECTION_MAP: dict[str, dict] = {
    # ---- 1. Scope -------------------------------------------------------
    "1.2": {
        "primary": ["Art.61.1", "Annex.XIV.A.1", "Annex.XIV.A.1.a"],
        "secondary": ["Art.61.12", "Annex.XIV.A.4"],
        "rationale": "Scope of the clinical evaluation is the clinical evaluation "
                     "plan obligation in Annex XIV Part A s1(a), under the overarching "
                     "duty in Article 61(1).",
    },

    # ---- 2. Device description -----------------------------------------
    "2.1": {
        "primary": ["Annex.II.1"],
        "secondary": ["Art.10.1", "Annex.I.23"],
        "rationale": "Device identification is the technical-documentation device "
                     "description required by Annex II s1.",
    },
    "2.2": {
        "primary": ["Art.10.1"],
        "secondary": ["Annex.II.1", "Annex.I.23"],
        "rationale": "Manufacturer identity is a general manufacturer obligation "
                     "(Art. 10) and a mandatory label particular (Annex I s23).",
    },
    "2.3": {
        "primary": ["Annex.II.1"],
        "secondary": ["Art.10.1"],
        "rationale": "Administrative particulars belong to the Annex II device "
                     "description and specification.",
    },
    "2.4": {
        "primary": ["Annex.II.1", "Annex.I.1"],
        "secondary": ["Annex.XIV.A.1.a"],
        "rationale": "Device description under Annex II s1; Annex I s1 fixes the "
                     "performance-as-intended requirement it must support.",
    },
    "2.5": {
        # Annex.VIII, not Annex.VIII.1. The rationale below always said "the
        # Annex VIII rules", but Annex.VIII.1 in the corpus is specifically
        # "DURATION OF USE". The rule that actually classifies a ureteral stent
        # is Annex.VIII.5 (invasive devices), so a retriever returning the
        # correct classification rule was being scored as wrong. The scope ID
        # covers the whole annex, which is what was meant.
        "primary": ["Art.51.1", "Annex.VIII"],
        "secondary": ["Art.52.1"],
        "rationale": "Classification is governed by Article 51 and the Annex VIII "
                     "rules; it determines the conformity assessment route.",
    },
    "2.6": {
        "primary": ["Annex.II.1", "Annex.I.1"],
        "secondary": ["Annex.I.10"],
        "rationale": "Technical specifications evidence the Annex I s1 performance "
                     "requirement and form part of the Annex II description.",
    },
    "2.7": {
        "primary": ["Annex.I.10", "Annex.II.1"],
        "secondary": ["Annex.I.11"],
        "rationale": "Components and materials engage Annex I s10 (chemical, physical "
                     "and biological properties); s11 applies because the stent is "
                     "supplied sterile.",
    },
    "2.8": {
        "primary": ["Annex.II.1", "Annex.I.1"],
        "secondary": ["Annex.XIV.A.1.a"],
        "rationale": "Mechanism of action is part of the device description and "
                     "underpins the claimed performance.",
    },
    "2.9": {
        "primary": ["Annex.II.1", "Annex.I.1"],
        "secondary": ["Annex.XIV.A.1.a", "Annex.I.23"],
        "rationale": "Intended purpose drives every downstream requirement; stated in "
                     "Annex II s1 and constrains Annex I s1 performance.",
    },
    "2.10": {
        "primary": ["Annex.I.23", "Annex.II.1"],
        "secondary": ["Annex.XIV.A.1.a"],
        "rationale": "Indications are mandatory information supplied with the device "
                     "under Annex I s23.",
    },
    "2.11": {
        "primary": ["Annex.I.23"],
        "secondary": ["Annex.I.8", "Annex.II.1"],
        "rationale": "Contraindications are an explicit Annex I s23 information "
                     "requirement; a blanket 'none known' also engages the s8 duty to "
                     "minimise and disclose known risks.",
    },
    "2.12": {
        "primary": ["Annex.XIV.A.1.a", "Annex.II.1"],
        "secondary": ["Annex.I.23"],
        "rationale": "Target population is a required element of the clinical "
                     "evaluation plan under Annex XIV Part A s1(a).",
    },
    "2.13": {
        "primary": ["Annex.I.5", "Annex.I.23"],
        "secondary": ["Annex.II.1"],
        "rationale": "Intended users bear on use-error risk (Annex I s5) and on the "
                     "information that must be supplied (s23).",
    },
    "2.14": {
        "primary": ["Annex.I.1", "Annex.XIV.A.1.a"],
        "secondary": ["Annex.I.8", "Art.61.1"],
        "rationale": "Clinical benefit must be specified in the clinical evaluation "
                     "plan and is one side of the Annex I benefit-risk determination.",
    },
    "2.15": {
        "primary": ["Art.7", "Annex.I.23"],
        "secondary": ["Annex.XIV.A.1.a"],
        "rationale": "Article 7 prohibits misleading claims; claims must be supported "
                     "by the clinical evaluation and reflected in the IFU.",
    },
    "2.17": {
        "primary": ["Annex.I.8", "Annex.I.23"],
        "secondary": ["Art.61.1", "Annex.XIV.A.1.a"],
        "rationale": "Undesirable side-effects must be minimised (Annex I s8) and "
                     "disclosed in the information supplied (s23).",
    },
    "2.18": {
        "primary": ["Art.10.9", "Annex.II.1"],
        "secondary": ["Annex.I.10"],
        "rationale": "Manufacturing process is covered by the quality management "
                     "system duty in Article 10(9) and the Annex II documentation.",
    },
    "2.19": {
        "primary": ["Art.10.9", "Annex.II.1"],
        "secondary": ["Art.61.11"],
        "rationale": "Design change history is a QMS obligation; material changes "
                     "trigger the Article 61(11) duty to update the clinical evaluation.",
    },

    # ---- 3. State of the art -------------------------------------------
    "3.1": {
        "primary": ["Annex.XIV.A.1.b", "Art.61.3.a"],
        "secondary": ["Annex.XIV.A.1.c"],
        "rationale": "Identification of pertinent data is Annex XIV Part A s1(b), and "
                     "the literature limb of the Article 61(3) procedure.",
    },
    "3.2": {
        "primary": ["Art.8.1", "Annex.II.1"],
        "secondary": ["Annex.I.1", "Art.9.1"],
        "rationale": "Harmonised standards (Art. 8) and common specifications (Art. 9) "
                     "give presumption of conformity with Annex I.",
    },
    "3.3": {
        "primary": ["Annex.XIV.A.1.a", "Art.61.3.a"],
        "secondary": ["Annex.XIV.A.2"],
        "rationale": "Clinical background establishes the state of the art the "
                     "evaluation must take into account (Annex XIV Part A s2).",
    },
    "3.4": {
        "primary": ["Annex.XIV.A.1.b"],
        "secondary": ["Annex.XIV.A.2"],
        "rationale": "Historical context is part of identifying and appraising "
                     "available clinical data.",
    },
    "3.5": {
        "primary": ["Annex.I.8", "Annex.I.3", "Art.61.1"],
        "secondary": ["Annex.I.4", "Annex.XIV.A.1.e"],
        "rationale": "Known failure modes and hazards are the core of the Annex I "
                     "risk management system (s3, s4) and the s8 minimisation duty.",
    },
    "3.6": {
        "primary": ["Art.61.3.c"],
        "secondary": ["Annex.XIV.A.2"],
        "rationale": "Article 61(3)(c) requires consideration of currently available "
                     "alternative treatment options.",
    },
    "3.7": {
        "primary": ["Art.61.5", "Annex.XIV.A.3"],
        "secondary": ["Art.61.4"],
        "rationale": "Equivalence is governed by Article 61(5) and the technical, "
                     "biological and clinical characteristics test in Annex XIV Part A s3.",
    },
    "3.8": {
        "primary": ["Annex.I.8", "Art.61.1"],
        "secondary": ["Annex.I.1", "Annex.I.2"],
        "rationale": "Acceptability of residual risk is the Annex I s8 judgement, "
                     "confirmed through clinical evaluation under Article 61(1).",
    },
    "3.9": {
        "primary": ["Annex.XIV.A.1.e", "Annex.XIV.A.2"],
        "secondary": ["Art.61.1"],
        "rationale": "Conclusions on the state of the art are the analysis limb of "
                     "Annex XIV Part A s1(e), which must be thorough and objective (s2).",
    },

    # ---- 4. Evaluation route and data ----------------------------------
    "4.1": {
        "primary": ["Art.61.1", "Annex.XIV.A.1"],
        "secondary": ["Art.61.4", "Art.61.10"],
        "rationale": "Choice of evaluation route follows from Article 61(1) and the "
                     "Annex XIV Part A process.",
    },
    "4.2": {
        "primary": ["Art.61.6", "Art.61.7"],
        "secondary": ["Art.61.4", "Art.61.8"],
        "rationale": "Well-established technology exemption from clinical "
                     "investigations sits in Article 61(6), with the justification "
                     "duty in 61(7).",
    },
    "4.2.1": {
        "primary": ["Art.61.5", "Annex.XIV.A.3"],
        "secondary": ["Art.61.4"],
        "rationale": "Equivalence conclusion is tested against Article 61(5) and "
                     "Annex XIV Part A s3.",
    },
    "4.2.2": {
        "primary": ["Art.52.1", "Annex.II.1"],
        "secondary": ["Art.61.1"],
        "rationale": "Conformity assessment summary maps to the Article 52 procedure "
                     "and the Annex II technical documentation.",
    },
    "4.3": {
        "primary": ["Annex.XIV.A.1.b", "Art.61.3.b"],
        "secondary": ["Annex.XIV.A.1.c"],
        "rationale": "Manufacturer-held data must be identified and appraised under "
                     "Annex XIV Part A s1(b)-(c).",
    },
    "4.3.1": {
        "primary": ["Annex.I.1", "Annex.II.1"],
        "secondary": ["Art.61.10"],
        "rationale": "Pre-clinical data evidences Annex I performance and safety where "
                     "clinical data alone are not relied on.",
    },
    "4.3.1.1": {
        "primary": ["Annex.I.10"],
        "secondary": ["Annex.I.11", "Annex.II.1"],
        "rationale": "Biocompatibility is squarely Annex I s10 (chemical, physical and "
                     "biological properties).",
    },
    "4.3.1.2": {
        "primary": ["Annex.I.1", "Annex.II.1"],
        "secondary": ["Art.61.10", "Annex.I.10"],
        "rationale": "Bench performance testing evidences the Annex I s1 requirement "
                     "that the device achieve its intended performance.",
    },
    "4.3.2.1": {
        "primary": ["Art.62.1", "Art.61.4"],
        "secondary": ["Annex.XIV.A.1.d", "Annex.XV.1"],
        "rationale": "Clinical investigations are governed by Article 62 and, for "
                     "implantables and class III, the Article 61(4) requirement.",
    },
    "4.3.2.2": {
        "primary": ["Art.83.1", "Art.84", "Annex.XIV.B.5"],
        "secondary": ["Art.85", "Art.86.1", "Art.61.11", "Annex.III"],
        "rationale": "PMS data engages the Article 83 PMS system, the Article 84 plan, "
                     "and PMCF under Annex XIV Part B, feeding back into Article 61(11).",
    },

    # ---- 4.5 Analysis ---------------------------------------------------
    "4.5": {
        "primary": ["Annex.XIV.A.1.e", "Art.61.3"],
        "secondary": ["Annex.XIV.A.2"],
        "rationale": "Data analysis is the Annex XIV Part A s1(e) obligation to reach "
                     "conclusions on safety, performance and benefit-risk.",
    },
    "4.5.1": {
        "primary": ["Annex.I.1", "Annex.I.8"],
        "secondary": ["Art.61.1"],
        "rationale": "Safety requirement maps to the Annex I s1 general obligation and "
                     "s8 risk minimisation.",
    },
    "4.5.2": {
        "primary": ["Annex.I.1", "Art.61.1", "Annex.XIV.A.1.e"],
        "secondary": ["Annex.I.8"],
        "rationale": "Overall safety and performance analysis is the central Article "
                     "61(1) confirmation of conformity with Annex I.",
    },
    "4.5.3": {
        "primary": ["Annex.I.1", "Annex.I.8", "Art.61.1"],
        "secondary": ["Annex.I.2", "Annex.XIV.A.1.e"],
        "rationale": "Benefit-risk acceptability is the Annex I s1 and s8 judgement "
                     "that Article 61(1) requires clinical evidence to support.",
    },
    "4.5.4": {
        "primary": ["Annex.I.8", "Art.61.1"],
        "secondary": ["Annex.I.2", "Annex.I.5"],
        "rationale": "Acceptability of side-effects is the Annex I s8 test, weighed "
                     "against benefit under Article 61(1).",
    },
}
# fmt: on

GRADE_PRIMARY = 2
GRADE_SECONDARY = 1


def referenced_clause_ids() -> set[str]:
    """Every clause ID this map points at, for validation against the corpus."""
    out: set[str] = set()
    for entry in SECTION_MAP.values():
        out.update(entry["primary"])
        out.update(entry["secondary"])
    return out


def graded_labels(section: str) -> dict[str, int]:
    """clause_id -> relevance grade for one CER section."""
    entry = SECTION_MAP.get(section)
    if not entry:
        return {}
    labels = {cid: GRADE_SECONDARY for cid in entry["secondary"]}
    labels.update({cid: GRADE_PRIMARY for cid in entry["primary"]})
    return labels

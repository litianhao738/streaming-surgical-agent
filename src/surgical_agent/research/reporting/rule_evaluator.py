"""Versioned, conservative English ontology/relationship matcher, not a clinical NLP model."""
import re

from .contracts import NAMES, TRIPLETS
from .event_report_generator import parse_report

RULE_VERSION = "gsr_rules_2"


def normalize(text):
    text = text.lower().replace("’", "'")
    for contracted, expanded in {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                                 "doesn't":"does not", "don't":"do not", "didn't":"did not", "hasn't":"has not", "haven't":"have not"}.items():
        text = re.sub(r"\b" + re.escape(contracted) + r"\b", expanded, text)
    return re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ")).strip()


ALIASES = {t: {name: {normalize(name)} for name in names} for t, names in NAMES.items()}
for verb, forms in {
    "grasp": "grasps grasped grasping", "retract": "retracts retracted retracting",
    "dissect": "dissects dissected dissecting", "coagulate": "coagulates coagulated coagulating",
    "clip": "clips clipped clipping", "cut": "cuts cutting", "aspirate": "aspirates aspirated aspirating",
    "irrigate": "irrigates irrigated irrigating", "pack": "packs packed packing",
}.items():
    ALIASES["verb"][verb].update(forms.split())
ALIASES["instrument"]["grasper"].add("graspers")
ALIASES["instrument"]["hook"].add("hooks")
ALIASES["instrument"]["clipper"].update(("clip applier", "clip applicator"))
ALIASES["instrument"]["irrigator"].add("suction irrigator")
ALIASES["target"]["blood_vessel"].add("blood vessels")
ALIASES["target"]["adhesion"].add("adhesions")
ALIASES["target"]["gut"].add("bowel")
ALIASES["phase"]["calot_triangle_dissection"].update(("calot's triangle dissection", "calot triangle dissection"))
RISK_TERMS = ("common bile duct", "common hepatic duct", "hepatic artery", "portal vein", "duodenum", "cbd")


def pattern(task, name=None):
    values = ALIASES[task][name] if name else set().union(*ALIASES[task].values())
    return "(?:" + "|".join(re.escape(x) for x in sorted(values, key=lambda x: (-len(x), x))) + ")"


def resolve(task, token):
    return next(name for name, values in ALIASES[task].items() if token in values)


def mentions(task, text):
    return {resolve(task, m.group()) for m in re.finditer(r"\b" + pattern(task) + r"\b", text)}


def _nonassertive_prefix(text, start):
    # Only the local clause prefix scopes these cues. Do not carry a negation
    # across a separate affirmative while/but/and clause.
    prefix = re.split(r"[.;,]|\b(?:while|but|and)\b", text[:start])[-1]
    return bool(re.search(r"\b(?:no|not|without|neither)\s+(?:the\s+)?$", prefix)
                or re.search(r"\b(?:if|unless|whether|possibly|perhaps)\b", prefix))


def extract(text):
    text = normalize(text)
    # Remove complete phase names first: 'clipping and cutting' is a phase,
    # not two additional observed interactions; gallbladder_dissection is not a target mention.
    phase_matches = list(re.finditer(r"\b" + pattern("phase") + r"\b", text))
    phases = {resolve("phase", m.group()) for m in phase_matches
              if not re.search(r"\b(?:not|no|without)\s+(?:in\s+|during\s+)?(?:the\s+)?$", text[max(0,m.start()-25):m.start()])}
    events = re.sub(r"\b" + pattern("phase") + r"\b", lambda m: " " * len(m.group()), text)
    inst, verb, target = pattern("instrument"), pattern("verb"), pattern("target")
    article = r"(?:(?:the|a|an)\s+)?"
    # Closed, non-negated auxiliary grammar. Avoid a wildcard here: it would
    # wrongly turn 'is not used to' and 'may be used to' into positive events.
    auxiliary = (r"(?:(?:is|are|was|were)\s+(?:currently\s+)?(?:being\s+)?(?:used|utilized)\s+(?:to|for)\s+"
                 r"|(?:has|have|had)\s+been\s+(?:used|utilized)\s+(?:to|for)\s+"
                 r"|(?:(?:is|are|was|were)\s+)?(?:actively\s+)?)")
    active = re.compile(r"\b(?P<i>" + inst + r")\s+" + auxiliary + r"(?P<v>" + verb + r")\s+" + article + r"(?P<t>" + target + r")\b")
    passive = re.compile(r"\b(?P<t>" + target + r")\s+(?:(?:is|are|was|were)\s+(?:being\s+)?|(?:has|have|had)\s+been\s+)(?P<v>" + verb + r")\s+by\s+" + article + r"(?P<i>" + inst + r")\b")
    join = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)"
    next_action = re.compile(join + r"(?P<v>" + verb + r")\s+" + article + r"(?P<t>" + target + r")\b")
    next_target = re.compile(join + article + r"(?P<t>" + target + r")\b")
    triples, spans = set(), []
    # These regexes require local directed relationships, never Cartesian products
    # of words appearing elsewhere in the report. Negated clauses cannot match.
    for expression in (active, passive):
        for match in expression.finditer(events):
            if _nonassertive_prefix(events, match.start()):
                continue
            triple = tuple(resolve(task, match.group(g)) for task, g in (("instrument", "i"), ("verb", "v"), ("target", "t")))
            triples.add(triple)
            spans.append(match.span())
            # In an active clause only, a following target or predicate can share
            # the explicit subject. Never carry context into a new sentence,
            # a new explicit instrument, negation, 'or', or a passive clause.
            if expression is active:
                end, current_i, current_v = match.end(), triple[0], triple[1]
                while True:
                    continuation = next_action.match(events, end)
                    if continuation:
                        current_v = resolve("verb", continuation.group("v"))
                    else:
                        continuation = next_target.match(events, end)
                    if continuation is None:
                        break
                    triples.add((current_i, current_v, resolve("target", continuation.group("t"))))
                    spans.append(continuation.span())
                    end = continuation.end()
    null_pattern = re.compile(r"\b(?P<i>" + inst + r")\s+(?:has|have) no identified action or target\b")
    for match in null_pattern.finditer(events):
        if _nonassertive_prefix(events, match.start()):
            continue
        triples.add((resolve("instrument", match.group("i")), "null_verb", "null_target"))
        spans.append(match.span())
    remaining = events
    for start, end in sorted(spans, reverse=True):
        remaining = remaining[:start] + " " * (end-start) + remaining[end:]
    # Known action words that cannot be resolved into directed relationships are
    # explicit unmatched claims, penalized rather than silently dropped.
    unmatched = [m.group() for m in re.finditer(r"\b" + pattern("verb") + r"\b", remaining)]
    # Resolved spans have explicit roles. Scanning them again as a word bag
    # would count specimen_bag (a target) as an additional instrument too.
    tokens = {(t, name) for t in ("instrument", "verb", "target") for name in mentions(t, remaining)}
    # The null idiom asserts null_verb/null_target even though these literal
    # tokens do not occur in natural language. They must face the same GT check.
    tokens |= {(task, name) for triple in triples for task, name in zip(("instrument", "verb", "target"), triple)}
    tokens |= {("phase", name) for name in phases}
    risks = {term for term in RISK_TERMS if re.search(r"\b" + re.escape(term) + r"\b", events)}
    return {"triples": triples, "phases": phases, "tokens": tokens, "risk_terms": risks,
            "unmatched_actions": unmatched, "normalized_text": text}


def evaluate_rule(raw_response, truth):
    if not truth.report_valid:
        return {"rule_score": None, "report_valid": False, "diagnostics": {"status": "masked"}}
    text, valid, format_status = parse_report(raw_response)
    parsed = extract(text)
    expected = {TRIPLETS[i] for i in truth.ivt_ids}
    predicted = parsed["triples"]
    tp, fp, fn = len(expected & predicted), len(predicted - expected), len(expected - predicted)
    fp += len(parsed["unmatched_actions"])
    den = 2 * tp + fp + fn
    explicit_empty = bool(re.search(r"\bno (?:surgical )?interaction(?:s)? (?:is|are) identified\b", parsed["normalized_text"]))
    ivt = 200 * tp / den if den else (100. if explicit_empty else 0.)
    phase = 100. if parsed["phases"] == {NAMES["phase"][truth.phase_id]} else 0.
    supported = {(task, name) for triple in expected for task, name in zip(("instrument", "verb", "target"), triple)}
    supported.add(("phase", NAMES["phase"][truth.phase_id]))
    unsupported = parsed["tokens"] - supported
    safety = 100. if text and not unsupported and not parsed["risk_terms"] else 0.
    score = .60 * ivt + .20 * phase + .15 * safety + .05 * (100. if valid else 0.)
    return {"rule_score": score, "report_valid": True, "diagnostics": {
        "rule_version": RULE_VERSION, "rule_ivt": ivt, "rule_phase": phase,
        "rule_hallucination": safety, "rule_format": 100. if valid else 0.,
        "format_status": format_status, "extracted_triples": sorted(predicted),
        "tp": tp, "fp": fp, "fn": fn, "unsupported_tokens": sorted(unsupported),
        "unsupported_anatomy": sorted(parsed["risk_terms"]), "unmatched_actions": parsed["unmatched_actions"],
        "parser_scope": "v2: explicit active/used-to/passive/null and local shared-subject coordination; conservative negation"}}

#!/usr/bin/env python3
"""Evaluation data generator for Antigence Subnet.

Generates labeled evaluation samples for any of the 4 domains (hallucination,
code_security, reasoning, bio) using parameterized templates or LLM API calls.
Produces both samples.json and manifest.json compatible with the EvaluationDataset
loader.

Supports two generation modes:
  - template (default): Rule-based generation, no API credentials required
  - llm: Uses LLM API for real-world outputs, requires --api-key or
    ANTIGENCE_LLM_API_KEY env var

Usage:
    python scripts/generate_eval_data.py --domain hallucination --count 150
    python scripts/generate_eval_data.py --domain all --count 150 --seed 42
    python scripts/generate_eval_data.py --domain bio --count 50 --append
    python scripts/generate_eval_data.py --domain all --count 50 --method llm --api-key sk-... --api-provider openai
    ANTIGENCE_LLM_API_KEY=sk-... python scripts/generate_eval_data.py --domain hallucination --count 20 --method llm
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Template pools -- parameterized data for generating diverse samples
# ---------------------------------------------------------------------------

# Hallucination domain pools
EVENTS = [
    ("the Battle of Hastings", "1066"),
    ("the signing of the Magna Carta", "1215"),
    ("the discovery of America by Columbus", "1492"),
    ("the start of the French Revolution", "1789"),
    ("the fall of the Berlin Wall", "1989"),
    ("the Moon landing", "1969"),
    ("the invention of the printing press", "1440"),
    ("the founding of Rome", "753 BC"),
    ("the end of the Cold War", "1991"),
    ("the start of World War I", "1914"),
    ("the assassination of Julius Caesar", "44 BC"),
    ("the Great Fire of London", "1666"),
    ("the American Declaration of Independence", "1776"),
    ("the start of the Renaissance", "14th century"),
    ("the invention of the telephone", "1876"),
]

CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Australia", "Canberra"),
    ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
    ("India", "New Delhi"), ("Mexico", "Mexico City"), ("Russia", "Moscow"),
    ("South Korea", "Seoul"), ("Argentina", "Buenos Aires"),
    ("Turkey", "Ankara"), ("Thailand", "Bangkok"), ("Sweden", "Stockholm"),
    ("Poland", "Warsaw"),
]

WRONG_CAPITALS = [
    "Sydney", "Lyon", "Osaka", "Rio de Janeiro", "Toronto",
    "Alexandria", "Mumbai", "Guadalajara", "St. Petersburg", "Busan",
    "Cordoba", "Istanbul", "Chiang Mai", "Gothenburg", "Krakow",
]

FAKE_AUTHORS = [
    "J.R. Henderson", "M.K. Patel", "S.A. Rodriguez", "L.T. Nguyen",
    "D.F. Schmidt", "P.W. Okonkwo", "R.C. Yamamoto", "A.B. Kowalski",
    "H.E. Lindqvist", "G.M. Fernandez", "K.J. O'Brien", "N.Q. Zhang",
]

FAKE_JOURNALS = [
    "Journal of Advanced Computational Science",
    "International Review of Applied Mathematics",
    "Proceedings of the Global AI Conference",
    "Annals of Theoretical Biology",
    "Quarterly Review of Data Science",
    "Archives of Neural Computing",
    "Frontiers in Algorithmic Research",
    "European Journal of Information Theory",
]

SCIENCE_FACTS_WRONG = [
    ("human body has how many bones", "The adult human body has 206 bones", "The adult human body has 312 bones"),
    ("largest organ in the human body", "The largest organ in the human body is the skin", "The largest organ in the human body is the liver"),
    ("boiling point of water at sea level", "Water boils at 100 degrees Celsius at sea level", "Water boils at 85 degrees Celsius at sea level"),
    ("speed of sound in air", "The speed of sound in air is approximately 343 m/s", "The speed of sound in air is approximately 520 m/s"),
    ("number of chromosomes in a human cell", "Human cells contain 46 chromosomes", "Human cells contain 52 chromosomes"),
    ("distance from Earth to the Moon", "The average distance from Earth to the Moon is about 384,400 km", "The average distance from Earth to the Moon is about 612,000 km"),
    ("chemical symbol for gold", "The chemical symbol for gold is Au", "The chemical symbol for gold is Gd"),
    ("pH of pure water", "The pH of pure water at 25C is 7.0", "The pH of pure water at 25C is 6.2"),
    ("freezing point of water", "Water freezes at 0 degrees Celsius", "Water freezes at -5 degrees Celsius"),
    ("number of planets in our solar system", "There are 8 planets in our solar system", "There are 11 planets in our solar system"),
]

CLAIM_TOPICS = [
    ("coffee consumption and longevity", "Some studies suggest moderate coffee consumption may be associated with reduced mortality risk",
     "Multiple meta-analyses conclusively prove that drinking 5+ cups of coffee daily guarantees a 40-year increase in lifespan"),
    ("exercise and mental health", "Regular exercise has been associated with improvements in mood and reduced symptoms of depression",
     "A single 10-minute walk completely eliminates all forms of clinical depression permanently"),
    ("meditation and stress", "Research suggests that regular meditation practice may help reduce perceived stress levels",
     "Meditation has been proven to reduce cortisol levels by 95% and completely prevent all stress-related diseases"),
    ("sleep duration", "Most adults require 7-9 hours of sleep for optimal health",
     "Scientists have definitively proven that exactly 4.5 hours of sleep maximizes cognitive performance"),
    ("vitamin C and colds", "The evidence for vitamin C preventing common colds is mixed and modest",
     "Taking 10g of vitamin C daily has been proven to make humans completely immune to all respiratory viruses"),
]

# Code security template pools
SQL_TABLES = ["users", "accounts", "products", "orders", "customers", "sessions", "payments", "logs"]
SQL_COLUMNS = ["id", "name", "email", "status", "created_at", "amount", "type", "role"]
FUNC_NAMES = ["get_record", "find_entry", "lookup_item", "fetch_data", "search_db", "query_table", "load_row", "retrieve_info"]
VAR_NAMES = ["user_input", "query_param", "search_term", "filter_val", "user_id", "item_name", "data_key", "record_ref"]

# Reasoning domain pools
SUBJECTS_A = ["birds", "mammals", "reptiles", "fish", "insects", "amphibians"]
SUBJECTS_B = ["penguins", "whales", "crocodiles", "salmon", "beetles", "frogs"]
PROPERTIES = ["can fly", "are warm-blooded", "have scales", "live in water", "have wings", "can regenerate limbs"]

MATH_PROBLEMS = [
    ("A store has {a} items. {b} are sold in the morning and {c} in the afternoon.", "{a}", "{b}", "{c}"),
    ("A farmer has {a} sheep. He buys {b} more and then sells {c}.", "{a}", "{b}", "{c}"),
    ("A tank holds {a} liters. {b} liters leak out and {c} liters are added.", "{a}", "{b}", "{c}"),
    ("A library has {a} books. {b} are borrowed and {c} are returned.", "{a}", "{b}", "{c}"),
]

# Bio domain pools
GENES = ["BRCA1", "TP53", "EGFR", "KRAS", "MYC", "APC", "RB1", "PTEN", "VHL", "CDH1", "MLH1", "BRAF"]
PROTEINS = ["hemoglobin", "insulin", "collagen", "keratin", "albumin", "actin", "myosin", "fibrinogen"]
MEASUREMENTS = ["pH", "temperature (C)", "concentration (mM)", "absorbance (OD)", "fold change", "expression level (TPM)"]
ORGANISMS = ["Homo sapiens", "Mus musculus", "Drosophila melanogaster", "Saccharomyces cerevisiae",
             "Caenorhabditis elegans", "Danio rerio", "Arabidopsis thaliana", "Escherichia coli"]

# ---------------------------------------------------------------------------
# Difficulty assignment
# ---------------------------------------------------------------------------

DIFFICULTY_WEIGHTS = {"easy": 0.30, "medium": 0.40, "hard": 0.30}


def assign_difficulty(rng: random.Random) -> str:
    """Assign difficulty with ~30/40/30 distribution."""
    r = rng.random()
    if r < 0.30:
        return "easy"
    elif r < 0.70:
        return "medium"
    else:
        return "hard"


def assign_honeypot(rng: random.Random) -> bool:
    """~13% chance of being a honeypot."""
    return rng.random() < 0.13


# ---------------------------------------------------------------------------
# Hallucination generators
# ---------------------------------------------------------------------------


def _gen_hall_factual_error(rng: random.Random) -> tuple[str, str]:
    """Generate a factual error hallucination sample."""
    choice = rng.randint(0, 2)
    if choice == 0:
        country, correct = rng.choice(CAPITALS)
        wrong = rng.choice(WRONG_CAPITALS)
        while wrong == correct:
            wrong = rng.choice(WRONG_CAPITALS)
        return (
            f"What is the capital of {country}?",
            f"The capital of {country} is {wrong}.",
        )
    elif choice == 1:
        event, correct_year = rng.choice(EVENTS)
        wrong_year = str(int(correct_year.split()[0]) + rng.choice([-3, -7, 5, 12, 20, -15])) if correct_year.isdigit() else str(rng.randint(1000, 2020))
        return (
            f"When did {event} occur?",
            f"{event.capitalize()} occurred in {wrong_year}.",
        )
    else:
        topic, correct_fact, wrong_fact = rng.choice(SCIENCE_FACTS_WRONG)
        return (
            f"Tell me about {topic}.",
            wrong_fact,
        )


def _gen_hall_fabricated_citation(rng: random.Random) -> tuple[str, str]:
    """Generate a fabricated citation sample."""
    author = rng.choice(FAKE_AUTHORS)
    journal = rng.choice(FAKE_JOURNALS)
    year = rng.randint(2018, 2025)
    topic = rng.choice(["neural network optimization", "protein folding prediction",
                        "climate modeling accuracy", "quantum error correction",
                        "natural language understanding", "drug interaction prediction"])
    return (
        f"What does recent research say about {topic}?",
        f"According to {author} et al. ({year}) in {journal}, "
        f"significant advances in {topic} have demonstrated a {rng.randint(30, 95)}% improvement "
        f"over previous benchmarks. This groundbreaking study of {rng.randint(500, 5000)} subjects "
        f"definitively established the superiority of the proposed approach.",
    )


def _gen_hall_hallucinated_fact(rng: random.Random) -> tuple[str, str]:
    """Generate a hallucinated fact sample (plausible but wrong)."""
    templates = [
        ("What is the national animal of {}?",
         "The national animal of {} is the {}."),
        ("What element has atomic number {}?",
         "The element with atomic number {} is {}."),
        ("Who invented the {}?",
         "The {} was invented by {} in {}."),
    ]
    choice = rng.randint(0, 2)
    if choice == 0:
        countries = ["Scotland", "Norway", "Brazil", "New Zealand", "Kenya"]
        animals = ["golden eagle", "arctic fox", "river dolphin", "kakapo", "black rhino"]
        wrong_animals = ["snow leopard", "mountain goat", "fire ant", "glass frog", "moon jellyfish"]
        idx = rng.randint(0, len(countries) - 1)
        return (
            f"What is the national animal of {countries[idx]}?",
            f"The national animal of {countries[idx]} is the {rng.choice(wrong_animals)}.",
        )
    elif choice == 1:
        numbers = [29, 47, 79, 82, 92]
        elements = ["Copper", "Silver", "Gold", "Lead", "Uranium"]
        wrong_elements = ["Titanium", "Platinum", "Bismuth", "Radium", "Thorium"]
        idx = rng.randint(0, len(numbers) - 1)
        return (
            f"What element has atomic number {numbers[idx]}?",
            f"The element with atomic number {numbers[idx]} is {rng.choice(wrong_elements)}.",
        )
    else:
        inventions = ["telescope", "microscope", "thermometer", "barometer", "stethoscope"]
        fake_inventors = ["Hans Lippershey", "Robert Hooke", "Daniel Fahrenheit", "Blaise Pascal", "Rene Laennec"]
        idx = rng.randint(0, len(inventions) - 1)
        fake_inv = rng.choice(FAKE_AUTHORS)
        return (
            f"Who invented the {inventions[idx]}?",
            f"The {inventions[idx]} was invented by {fake_inv} in {rng.randint(1500, 1900)}.",
        )


def _gen_hall_unsupported_claim(rng: random.Random) -> tuple[str, str]:
    """Generate an unsupported claim sample."""
    topic, modest_claim, overblown_claim = rng.choice(CLAIM_TOPICS)
    return (
        f"What does the research say about {topic}?",
        overblown_claim,
    )


def _gen_hall_normal(rng: random.Random) -> tuple[str, str]:
    """Generate a normal (correct) hallucination domain sample."""
    choice = rng.randint(0, 2)
    if choice == 0:
        country, capital = rng.choice(CAPITALS)
        return (
            f"What is the capital of {country}?",
            f"The capital of {country} is {capital}.",
        )
    elif choice == 1:
        event, year = rng.choice(EVENTS)
        return (
            f"When did {event} occur?",
            f"{event.capitalize()} occurred in {year}.",
        )
    else:
        topic, correct_fact, _ = rng.choice(SCIENCE_FACTS_WRONG)
        return (
            f"Tell me about {topic}.",
            correct_fact,
        )


HALL_ANOMALY_TYPES = ["factual_error", "fabricated_citation", "hallucinated_fact", "unsupported_claim"]
HALL_ANOMALY_GENERATORS = {
    "factual_error": _gen_hall_factual_error,
    "fabricated_citation": _gen_hall_fabricated_citation,
    "hallucinated_fact": _gen_hall_hallucinated_fact,
    "unsupported_claim": _gen_hall_unsupported_claim,
}


def generate_hallucination_samples(
    n: int, start_id: int, seed: int = 42
) -> tuple[list[dict], dict]:
    """Generate n hallucination domain samples.

    Args:
        n: Number of samples to generate.
        start_id: Starting ID number (e.g., 61 for eval-hall-061).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (samples_list, manifest_dict).
    """
    rng = random.Random(seed)
    samples = []
    manifest = {}

    # Target ~50% anomalous
    n_anomalous = int(n * 0.50)
    labels = ["anomalous"] * n_anomalous + ["normal"] * (n - n_anomalous)
    rng.shuffle(labels)

    for i, label in enumerate(labels):
        sample_id = f"eval-hall-{start_id + i:03d}"
        difficulty = assign_difficulty(rng)
        is_honeypot = assign_honeypot(rng) if label == "anomalous" else False

        if label == "anomalous":
            atype = rng.choice(HALL_ANOMALY_TYPES)
            prompt, output = HALL_ANOMALY_GENERATORS[atype](rng)
            manifest[sample_id] = {
                "ground_truth_label": "anomalous",
                "ground_truth_type": atype,
                "is_honeypot": is_honeypot,
            }
        else:
            prompt, output = _gen_hall_normal(rng)
            manifest[sample_id] = {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }

        samples.append({
            "id": sample_id,
            "prompt": prompt,
            "output": output,
            "domain": "hallucination",
            "metadata": {"difficulty": difficulty, "source": "template"},
        })

    return samples, manifest


# ---------------------------------------------------------------------------
# Code Security generators
# ---------------------------------------------------------------------------

CS_ANOMALY_TYPES = [
    "sql_injection", "xss", "command_injection", "path_traversal",
    "hardcoded_credentials", "code_injection", "weak_crypto",
]


def _gen_cs_sql_injection(rng: random.Random) -> tuple[str, str, str]:
    """Generate SQL injection vulnerable code."""
    table = rng.choice(SQL_TABLES)
    col = rng.choice(SQL_COLUMNS)
    func = rng.choice(FUNC_NAMES)
    var = rng.choice(VAR_NAMES)
    return (
        f"Write a function to query {table} by {col}",
        "",
        f"def {func}({var}):\n"
        f"    query = \"SELECT * FROM {table} WHERE {col}='\" + {var} + \"'\"\n"
        f"    cursor.execute(query)\n"
        f"    return cursor.fetchall()",
    )


def _gen_cs_xss(rng: random.Random) -> tuple[str, str, str]:
    """Generate XSS vulnerable code."""
    func = rng.choice(FUNC_NAMES)
    var = rng.choice(VAR_NAMES)
    return (
        f"Write a function to render user content on a page",
        "",
        f"def {func}({var}):\n"
        f"    return f\"<div>{{{{  {var} }}}}</div>\"\n",
    )


def _gen_cs_command_injection(rng: random.Random) -> tuple[str, str, str]:
    """Generate command injection vulnerable code."""
    func = rng.choice(FUNC_NAMES)
    var = rng.choice(VAR_NAMES)
    cmds = ["ls", "cat", "ping", "nslookup", "find"]
    cmd = rng.choice(cmds)
    return (
        f"Write a function to run a system {cmd} command",
        "",
        f"import os\n\n"
        f"def {func}({var}):\n"
        f"    return os.popen('{cmd} ' + {var}).read()",
    )


def _gen_cs_path_traversal(rng: random.Random) -> tuple[str, str, str]:
    """Generate path traversal vulnerable code."""
    func = rng.choice(FUNC_NAMES)
    var = rng.choice(VAR_NAMES)
    return (
        f"Write a function to read a file by name",
        "",
        f"def {func}({var}):\n"
        f"    with open('/data/uploads/' + {var}) as f:\n"
        f"        return f.read()",
    )


def _gen_cs_hardcoded_creds(rng: random.Random) -> tuple[str, str, str]:
    """Generate hardcoded credentials code."""
    func = rng.choice(FUNC_NAMES)
    passwords = ["admin123", "password1", "s3cr3t!", "P@ssw0rd", "root_pass"]
    return (
        f"Write a function to connect to the database",
        "",
        f"def {func}():\n"
        f"    db_user = 'admin'\n"
        f"    db_pass = '{rng.choice(passwords)}'\n"
        f"    return connect(host='db.internal', user=db_user, password=db_pass)",
    )


def _gen_cs_code_injection(rng: random.Random) -> tuple[str, str, str]:
    """Generate code injection (eval) vulnerable code."""
    func = rng.choice(FUNC_NAMES)
    var = rng.choice(VAR_NAMES)
    return (
        f"Write a function to evaluate a math expression",
        "",
        f"def {func}({var}):\n"
        f"    return eval({var})",
    )


def _gen_cs_weak_crypto(rng: random.Random) -> tuple[str, str, str]:
    """Generate weak cryptography code."""
    func = rng.choice(FUNC_NAMES)
    var = rng.choice(VAR_NAMES)
    algos = ["md5", "sha1"]
    algo = rng.choice(algos)
    return (
        f"Write a function to hash a password",
        "",
        f"import hashlib\n\n"
        f"def {func}({var}):\n"
        f"    return hashlib.{algo}({var}.encode()).hexdigest()",
    )


CS_ANOMALY_GENERATORS = {
    "sql_injection": _gen_cs_sql_injection,
    "xss": _gen_cs_xss,
    "command_injection": _gen_cs_command_injection,
    "path_traversal": _gen_cs_path_traversal,
    "hardcoded_credentials": _gen_cs_hardcoded_creds,
    "code_injection": _gen_cs_code_injection,
    "weak_crypto": _gen_cs_weak_crypto,
}


def _gen_cs_normal(rng: random.Random) -> tuple[str, str, str]:
    """Generate secure (normal) code sample."""
    choice = rng.randint(0, 3)
    table = rng.choice(SQL_TABLES)
    col = rng.choice(SQL_COLUMNS)
    func = rng.choice(FUNC_NAMES)
    var = rng.choice(VAR_NAMES)

    if choice == 0:
        return (
            f"Write a function to query {table} by {col}",
            "",
            f"def {func}({var}):\n"
            f"    cursor.execute('SELECT * FROM {table} WHERE {col} = %s', ({var},))\n"
            f"    return cursor.fetchall()",
        )
    elif choice == 1:
        return (
            f"Write a function to run a system command safely",
            "",
            f"import subprocess\n\n"
            f"def {func}({var}):\n"
            f"    result = subprocess.run([{var}], capture_output=True, text=True, shell=False)\n"
            f"    return result.stdout",
        )
    elif choice == 2:
        return (
            f"Write a function to read a file safely",
            "",
            f"import os\n\n"
            f"def {func}({var}):\n"
            f"    safe_path = os.path.realpath(os.path.join('/data/uploads', {var}))\n"
            f"    if not safe_path.startswith('/data/uploads'):\n"
            f"        raise ValueError('Path traversal detected')\n"
            f"    with open(safe_path) as f:\n"
            f"        return f.read()",
        )
    else:
        return (
            f"Write a function to hash a password securely",
            "",
            f"import hashlib\nimport secrets\n\n"
            f"def {func}({var}):\n"
            f"    salt = secrets.token_hex(16)\n"
            f"    hashed = hashlib.pbkdf2_hmac('sha256', {var}.encode(), salt.encode(), 100000)\n"
            f"    return salt + ':' + hashed.hex()",
        )


def generate_code_security_samples(
    n: int, start_id: int, seed: int = 42
) -> tuple[list[dict], dict]:
    """Generate n code_security domain samples."""
    rng = random.Random(seed)
    samples = []
    manifest = {}

    n_anomalous = int(n * 0.50)
    labels = ["anomalous"] * n_anomalous + ["normal"] * (n - n_anomalous)
    rng.shuffle(labels)

    for i, label in enumerate(labels):
        sample_id = f"eval-cs-{start_id + i:03d}"
        difficulty = assign_difficulty(rng)
        is_honeypot = assign_honeypot(rng) if label == "anomalous" else False

        if label == "anomalous":
            atype = rng.choice(CS_ANOMALY_TYPES)
            prompt, output, code = CS_ANOMALY_GENERATORS[atype](rng)
            manifest[sample_id] = {
                "ground_truth_label": "anomalous",
                "ground_truth_type": atype,
                "is_honeypot": is_honeypot,
            }
        else:
            prompt, output, code = _gen_cs_normal(rng)
            manifest[sample_id] = {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }

        samples.append({
            "id": sample_id,
            "prompt": prompt,
            "output": output,
            "code": code,
            "domain": "code_security",
            "metadata": {"difficulty": difficulty, "source": "template"},
        })

    return samples, manifest


# ---------------------------------------------------------------------------
# Reasoning generators
# ---------------------------------------------------------------------------

REA_ANOMALY_TYPES = [
    "logical_contradiction", "non_sequitur", "constraint_violation",
    "circular_reasoning", "arithmetic_error", "logical_fallacy",
]


def _gen_rea_logical_contradiction(rng: random.Random) -> tuple[str, str]:
    """Generate a logical contradiction reasoning sample."""
    idx = rng.randint(0, len(SUBJECTS_A) - 1)
    subj_a = SUBJECTS_A[idx]
    subj_b = SUBJECTS_B[idx]
    prop = PROPERTIES[idx]
    return (
        f"If all {subj_a} {prop} and {subj_b} are {subj_a}, can {subj_b} {prop.replace('are ', '').replace('have ', 'have ').replace('can ', '')}?",
        f"Step 1: All {subj_a} {prop}. "
        f"Step 2: {subj_b.capitalize()} are {subj_a}. "
        f"Step 3: Therefore, {subj_b} {prop}. "
        f"However, we know {subj_b} are not {subj_a}. "
        f"Since {subj_b} are not {subj_a}, the conclusion is invalid.",
    )


def _gen_rea_non_sequitur(rng: random.Random) -> tuple[str, str]:
    """Generate a non-sequitur reasoning sample."""
    premises = [
        ("The sun is a star", "Stars emit light", "Therefore, bananas are yellow"),
        ("All cats are animals", "Some animals can swim", "Therefore, the Earth is flat"),
        ("Iron is a metal", "Metals conduct electricity", "Therefore, plants need water"),
        ("Triangles have three sides", "Squares have four sides", "Therefore, music is beautiful"),
    ]
    p1, p2, conclusion = rng.choice(premises)
    return (
        f"Evaluate the reasoning: {p1}. {p2}. {conclusion}.",
        f"Step 1: {p1}. Step 2: {p2}. Step 3: {conclusion}. This follows logically from the premises.",
    )


def _gen_rea_arithmetic_error(rng: random.Random) -> tuple[str, str]:
    """Generate an arithmetic error reasoning sample."""
    template, _, _, _ = rng.choice(MATH_PROBLEMS)
    a = rng.randint(50, 200)
    b = rng.randint(10, a // 2)
    c = rng.randint(5, a // 3)
    correct = a - b + c if "buys" in template or "added" in template or "returned" in template else a - b - c
    wrong = correct + rng.choice([-15, -10, 10, 15, 20])
    problem = template.format(a=a, b=b, c=c)
    return (
        problem,
        f"Starting with {a}. After removing {b}, we have {a - b}. "
        f"After the next operation with {c}, the result is {wrong}.",
    )


def _gen_rea_circular_reasoning(rng: random.Random) -> tuple[str, str]:
    """Generate circular reasoning sample."""
    examples = [
        ("Why is this law just?", "This law is just because it is the law, and all laws are just because they are laws."),
        ("Why should we trust this source?", "We should trust this source because it is reliable, and we know it is reliable because we trust it."),
        ("Why is this the best approach?", "This is the best approach because it produces the best results, and the best results come from the best approach."),
        ("Why is this statement true?", "This statement is true because it is correct, and it is correct because the statement is true."),
    ]
    prompt, output = rng.choice(examples)
    return prompt, output


def _gen_rea_logical_fallacy(rng: random.Random) -> tuple[str, str]:
    """Generate a logical fallacy sample."""
    fallacies = [
        ("Is it valid to say: Everyone uses this product, so it must be good?",
         "Everyone uses this product, so it must be good. The popularity of a product is direct proof of its quality."),
        ("Evaluate: You either support this policy or you hate the country.",
         "There are only two options: support the policy or hate the country. Since you don't support it, you must hate the country."),
        ("Is this argument valid: My grandfather smoked and lived to 95, so smoking is safe?",
         "My grandfather smoked and lived to 95. This proves smoking is safe for everyone."),
        ("Evaluate: Expert X has a PhD, therefore everything they say about any topic is true.",
         "Expert X has a PhD in physics. Therefore, their opinion on economics must be correct."),
    ]
    prompt, output = rng.choice(fallacies)
    return prompt, output


def _gen_rea_constraint_violation(rng: random.Random) -> tuple[str, str]:
    """Generate a constraint violation reasoning sample."""
    constraints = [
        ("A meeting room fits 10 people. We need to seat 15 people.",
         "We can fit all 15 people in the meeting room. Step 1: The room fits 10. Step 2: We add 5 more chairs. Step 3: All 15 are comfortably seated within capacity."),
        ("A budget is $1000. Items cost $400, $500, and $300.",
         "We can buy all three items within budget. Item 1: $400. Item 2: $500. Item 3: $300. Total: $1200. This is within the $1000 budget."),
        ("A train departs at 3pm and arrives at 2pm the same day.",
         "The train departs at 3pm and arrives at 2pm on the same day. The journey takes -1 hours, which is valid for express routes."),
    ]
    prompt, output = rng.choice(constraints)
    return prompt, output


REA_ANOMALY_GENERATORS = {
    "logical_contradiction": _gen_rea_logical_contradiction,
    "non_sequitur": _gen_rea_non_sequitur,
    "constraint_violation": _gen_rea_constraint_violation,
    "circular_reasoning": _gen_rea_circular_reasoning,
    "arithmetic_error": _gen_rea_arithmetic_error,
    "logical_fallacy": _gen_rea_logical_fallacy,
}


def _gen_rea_normal(rng: random.Random) -> tuple[str, str]:
    """Generate a normal (valid) reasoning sample."""
    choice = rng.randint(0, 2)
    if choice == 0:
        idx = rng.randint(0, len(SUBJECTS_A) - 1)
        subj_a = SUBJECTS_A[idx]
        subj_b = SUBJECTS_B[idx]
        prop = PROPERTIES[idx]
        return (
            f"If all {subj_a} {prop} and {subj_b} are {subj_a}, can {subj_b} {prop.replace('are ', '').replace('have ', 'have ').replace('can ', '')}?",
            f"Step 1: All {subj_a} {prop}. "
            f"Step 2: {subj_b.capitalize()} are {subj_a}. "
            f"Step 3: By modus ponens, {subj_b} {prop}. "
            f"The conclusion follows validly from the premises.",
        )
    elif choice == 1:
        a = rng.randint(50, 200)
        b = rng.randint(10, a // 2)
        c = rng.randint(5, a // 3)
        result = a - b - c
        return (
            f"A store has {a} items. {b} are sold in the morning and {c} in the afternoon. How many remain?",
            f"Step 1: Start with {a}. Step 2: Subtract {b} sold in the morning: {a} - {b} = {a - b}. "
            f"Step 3: Subtract {c} sold in the afternoon: {a - b} - {c} = {result}. "
            f"Answer: {result} items remain.",
        )
    else:
        return (
            "Is the following valid: All mammals are warm-blooded. Dogs are mammals. Therefore dogs are warm-blooded.",
            "Step 1: Major premise: All mammals are warm-blooded. "
            "Step 2: Minor premise: Dogs are mammals. "
            "Step 3: By modus ponens, dogs are warm-blooded. "
            "The argument is a valid syllogism.",
        )


def generate_reasoning_samples(
    n: int, start_id: int, seed: int = 42
) -> tuple[list[dict], dict]:
    """Generate n reasoning domain samples."""
    rng = random.Random(seed)
    samples = []
    manifest = {}

    n_anomalous = int(n * 0.50)
    labels = ["anomalous"] * n_anomalous + ["normal"] * (n - n_anomalous)
    rng.shuffle(labels)

    for i, label in enumerate(labels):
        sample_id = f"eval-rea-{start_id + i:03d}"
        difficulty = assign_difficulty(rng)
        is_honeypot = assign_honeypot(rng) if label == "anomalous" else False

        if label == "anomalous":
            atype = rng.choice(REA_ANOMALY_TYPES)
            prompt, output = REA_ANOMALY_GENERATORS[atype](rng)
            manifest[sample_id] = {
                "ground_truth_label": "anomalous",
                "ground_truth_type": atype,
                "is_honeypot": is_honeypot,
            }
        else:
            prompt, output = _gen_rea_normal(rng)
            manifest[sample_id] = {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }

        samples.append({
            "id": sample_id,
            "prompt": prompt,
            "output": output,
            "domain": "reasoning",
            "metadata": {"difficulty": difficulty, "source": "template"},
        })

    return samples, manifest


# ---------------------------------------------------------------------------
# Bio generators
# ---------------------------------------------------------------------------

BIO_ANOMALY_TYPES = [
    "value_out_of_range", "statistical_anomaly", "physical_impossibility",
    "species_mismatch", "unit_inconsistency", "p_value_fabrication",
    "arithmetic_error",
]


def _gen_bio_value_out_of_range(rng: random.Random) -> tuple[str, str]:
    """Generate value-out-of-range bio sample."""
    templates = [
        ("Report the pH measurements.", f"Buffer A: pH 7.4, Buffer B: pH {rng.uniform(15, 20):.1f}, Buffer C: pH 6.8. Buffer B shows pH outside measurable range."),
        ("Report the temperature readings.", f"Incubator 1: 37C, Incubator 2: {rng.randint(500, 5000)}C, Incubator 3: 25C. Incubator 2 temperature is impossibly high."),
        ("Report the concentration measurements.", f"Well 1: {rng.uniform(-50, -1):.1f} mM, Well 2: 0.5 mM, Well 3: 1.0 mM. Well 1 shows a negative concentration."),
    ]
    prompt, output = rng.choice(templates)
    return prompt, output


def _gen_bio_statistical_anomaly(rng: random.Random) -> tuple[str, str]:
    """Generate statistical anomaly bio sample."""
    gene = rng.choice(GENES)
    fold = rng.uniform(500, 2000)
    return (
        f"Report the fold change for {gene} expression.",
        f"{gene} showed a fold change of {fold:.1f} between treatment and control groups. "
        f"This is biologically implausible as typical fold changes for {gene} range from 0.5 to 10.",
    )


def _gen_bio_physical_impossibility(rng: random.Random) -> tuple[str, str]:
    """Generate physical impossibility bio sample."""
    templates = [
        ("Report the cell doubling time.", f"The cell line showed a doubling time of {rng.uniform(-10, -1):.1f} hours, indicating cells were dividing in reverse."),
        ("Report the protein molecular weight.", f"The protein has a molecular weight of {rng.randint(-500, -10)} kDa, which is negative."),
        ("Report the enzyme reaction velocity.", f"The enzyme showed Vmax of {rng.uniform(-100, -5):.1f} umol/min, a negative reaction rate."),
    ]
    prompt, output = rng.choice(templates)
    return prompt, output


def _gen_bio_species_mismatch(rng: random.Random) -> tuple[str, str]:
    """Generate species mismatch bio sample."""
    org1 = rng.choice(ORGANISMS[:4])
    org2 = rng.choice(ORGANISMS[4:])
    gene = rng.choice(GENES)
    return (
        f"Describe the {gene} knockout phenotype in {org1}.",
        f"The {gene} knockout in {org1} was performed using CRISPR-Cas9. "
        f"The phenotype matched previously described {gene} knockouts in {org2}, "
        f"confirming cross-species conservation. However, the protocol used {org2}-specific primers on {org1} tissue.",
    )


def _gen_bio_unit_inconsistency(rng: random.Random) -> tuple[str, str]:
    """Generate unit inconsistency bio sample."""
    meas = rng.choice(MEASUREMENTS)
    protein = rng.choice(PROTEINS)
    return (
        f"Report the {protein} {meas} measurements.",
        f"{protein} sample A: 5.2 mg/mL, sample B: 3100 ug/dL, sample C: 0.8 mol/L. "
        f"The units are inconsistent across samples and one value appears to be in wrong units.",
    )


def _gen_bio_p_value_fabrication(rng: random.Random) -> tuple[str, str]:
    """Generate p-value fabrication bio sample."""
    comparisons = rng.randint(3, 6)
    p_values = [round(rng.uniform(0.001, 0.049), 4) for _ in range(comparisons)]
    return (
        f"Report the statistical significance of {comparisons} comparisons.",
        f"All {comparisons} comparisons were significant: "
        + ", ".join(f"p={p}" for p in p_values) +
        f". The probability of all {comparisons} being independently significant at p<0.05 "
        f"without correction is {0.05**comparisons:.2e}, suggesting possible p-hacking.",
    )


def _gen_bio_arithmetic_error(rng: random.Random) -> tuple[str, str]:
    """Generate arithmetic error bio sample."""
    n_total = rng.randint(100, 500)
    group_a = rng.randint(30, n_total // 2)
    group_b = n_total - group_a
    wrong_total = n_total + rng.choice([10, -15, 20, -25])
    return (
        f"Report the sample distribution across groups.",
        f"Group A: n={group_a}, Group B: n={group_b}. "
        f"Total sample size: n={wrong_total}. "
        f"The sum of group sizes ({group_a} + {group_b} = {group_a + group_b}) does not match the reported total ({wrong_total}).",
    )


BIO_ANOMALY_GENERATORS = {
    "value_out_of_range": _gen_bio_value_out_of_range,
    "statistical_anomaly": _gen_bio_statistical_anomaly,
    "physical_impossibility": _gen_bio_physical_impossibility,
    "species_mismatch": _gen_bio_species_mismatch,
    "unit_inconsistency": _gen_bio_unit_inconsistency,
    "p_value_fabrication": _gen_bio_p_value_fabrication,
    "arithmetic_error": _gen_bio_arithmetic_error,
}


def _gen_bio_normal(rng: random.Random) -> tuple[str, str]:
    """Generate normal (valid) bio sample."""
    choice = rng.randint(0, 3)
    if choice == 0:
        gene = rng.choice(GENES)
        fold = round(rng.uniform(0.5, 5.0), 2)
        return (
            f"Report the {gene} expression results.",
            f"{gene} showed a fold change of {fold} (p=0.03) between treatment and control. "
            f"This is consistent with previously reported {gene} regulation in this pathway.",
        )
    elif choice == 1:
        protein = rng.choice(PROTEINS)
        conc = round(rng.uniform(0.1, 10.0), 2)
        return (
            f"Report the {protein} quantification results.",
            f"{protein} concentration was measured at {conc} mg/mL across triplicates "
            f"(CV={rng.uniform(2, 8):.1f}%). This is within the expected range for this assay.",
        )
    elif choice == 2:
        return (
            "Report the pH and temperature conditions.",
            f"Buffer pH was maintained at {round(rng.uniform(6.8, 7.6), 1)}. "
            f"Incubation temperature was {rng.choice([25, 30, 37])}C. "
            f"All conditions were within standard protocol ranges.",
        )
    else:
        n_total = rng.randint(100, 500)
        group_a = rng.randint(40, n_total // 2)
        group_b = n_total - group_a
        return (
            "Report the sample distribution.",
            f"Total sample size: n={n_total}. "
            f"Group A: n={group_a}, Group B: n={group_b}. "
            f"Groups were balanced with no significant demographic differences (p=0.72).",
        )


def generate_bio_samples(
    n: int, start_id: int, seed: int = 42
) -> tuple[list[dict], dict]:
    """Generate n bio domain samples."""
    rng = random.Random(seed)
    samples = []
    manifest = {}

    n_anomalous = int(n * 0.50)
    labels = ["anomalous"] * n_anomalous + ["normal"] * (n - n_anomalous)
    rng.shuffle(labels)

    for i, label in enumerate(labels):
        sample_id = f"eval-bio-{start_id + i:03d}"
        difficulty = assign_difficulty(rng)
        is_honeypot = assign_honeypot(rng) if label == "anomalous" else False

        if label == "anomalous":
            atype = rng.choice(BIO_ANOMALY_TYPES)
            prompt, output = BIO_ANOMALY_GENERATORS[atype](rng)
            manifest[sample_id] = {
                "ground_truth_label": "anomalous",
                "ground_truth_type": atype,
                "is_honeypot": is_honeypot,
            }
        else:
            prompt, output = _gen_bio_normal(rng)
            manifest[sample_id] = {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }

        samples.append({
            "id": sample_id,
            "prompt": prompt,
            "output": output,
            "domain": "bio",
            "metadata": {"difficulty": difficulty, "source": "template"},
        })

    return samples, manifest


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "hallucination": generate_hallucination_samples,
    "code_security": generate_code_security_samples,
    "reasoning": generate_reasoning_samples,
    "bio": generate_bio_samples,
}

DOMAIN_PREFIXES = {
    "hallucination": "eval-hall",
    "code_security": "eval-cs",
    "reasoning": "eval-rea",
    "bio": "eval-bio",
}


# ---------------------------------------------------------------------------
# LLM generation mode -- optional, requires API credentials
# ---------------------------------------------------------------------------

LLM_PROMPTS = {
    "hallucination": {
        "anomalous": (
            "Generate a factually incorrect response to the following question. "
            "The response should contain a subtle factual error that could be "
            "mistaken as correct. Question: {prompt}\n"
            "Respond with ONLY the incorrect answer, no explanation."
        ),
        "normal": (
            "Answer the following question accurately and concisely. "
            "Question: {prompt}\nRespond with ONLY the answer."
        ),
    },
    "code_security": {
        "anomalous": (
            "Write a Python function that {description}. Include a subtle "
            "{vuln_type} vulnerability in the code. Return ONLY the code, "
            "no explanation."
        ),
        "normal": (
            "Write a secure Python function that {description}. Follow "
            "security best practices. Return ONLY the code, no explanation."
        ),
    },
    "reasoning": {
        "anomalous": (
            "Provide a logical argument about {topic} that contains a subtle "
            "logical fallacy ({fallacy_type}). Present it as if it were valid "
            "reasoning. Return ONLY the argument."
        ),
        "normal": (
            "Provide a valid logical argument about {topic}. Ensure the "
            "reasoning is sound. Return ONLY the argument."
        ),
    },
    "bio": {
        "anomalous": (
            "Describe a biological finding about {gene_or_protein} that "
            "contains a subtle scientific error (wrong measurement, incorrect "
            "pathway, fabricated value). Present it as if it were from a "
            "research paper. Return ONLY the description."
        ),
        "normal": (
            "Describe a plausible biological finding about {gene_or_protein} "
            "with realistic measurements and correct pathway information. "
            "Return ONLY the description."
        ),
    },
}


def _call_llm_api(
    prompt: str,
    api_key: str,
    api_provider: str,
    model: str = "gpt-4o-mini",
) -> str | None:
    """Call an LLM API and return the response text, or None on failure.

    Supports OpenAI-compatible, Anthropic, and local (Ollama) providers.
    Uses stdlib urllib.request only -- no external dependencies.

    Args:
        prompt: The prompt to send to the LLM.
        api_key: API key for authentication.
        api_provider: One of 'openai', 'anthropic', 'local'.
        model: Model name (default: gpt-4o-mini for OpenAI).

    Returns:
        Response text string, or None on any error.
    """
    try:
        if api_provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 512,
            }).encode()
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]

        elif api_provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
            body = json.dumps({
                "model": "claude-3-haiku-20240307",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return data["content"][0]["text"]

        elif api_provider == "local":
            url = "http://localhost:11434/api/generate"
            headers = {"Content-Type": "application/json"}
            body = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
            }).encode()
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return data.get("response", "")

        else:
            print(f"[WARNING] Unknown API provider: {api_provider}")
            return None

    except Exception as e:
        print(f"[WARNING] LLM API call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM generators -- per-domain, same signature pattern as template generators
# ---------------------------------------------------------------------------


def generate_hallucination_samples_llm(
    n: int,
    start_id: int,
    seed: int = 42,
    api_key: str = "",
    api_provider: str = "openai",
) -> tuple[list[dict], dict]:
    """Generate n hallucination samples using LLM API.

    Falls back to template generation per-sample on API failure.
    """
    rng = random.Random(seed)
    samples = []
    manifest = {}

    n_anomalous = int(n * 0.50)
    labels = ["anomalous"] * n_anomalous + ["normal"] * (n - n_anomalous)
    rng.shuffle(labels)

    for i, label in enumerate(labels):
        sample_id = f"eval-hall-{start_id + i:03d}"
        difficulty = assign_difficulty(rng)
        is_honeypot = assign_honeypot(rng) if label == "anomalous" else False

        # Select a base prompt from template pool
        base_prompt_data = rng.choice(EVENTS + [(c, cap) for c, cap in CAPITALS])
        base_question = f"What do you know about {base_prompt_data[0]}?"

        # Try LLM generation
        llm_prompt = LLM_PROMPTS["hallucination"][label].format(prompt=base_question)
        llm_response = _call_llm_api(llm_prompt, api_key, api_provider)

        if llm_response is not None:
            source = "llm"
            prompt_text = base_question
            output_text = llm_response
        else:
            # Fallback to template
            print(f"[WARNING] LLM fallback for {sample_id}, using template")
            source = "template"
            if label == "anomalous":
                atype = rng.choice(HALL_ANOMALY_TYPES)
                prompt_text, output_text = HALL_ANOMALY_GENERATORS[atype](rng)
            else:
                prompt_text, output_text = _gen_hall_normal(rng)

        if label == "anomalous":
            atype_for_manifest = rng.choice(HALL_ANOMALY_TYPES) if llm_response is not None else atype
            manifest[sample_id] = {
                "ground_truth_label": "anomalous",
                "ground_truth_type": atype_for_manifest,
                "is_honeypot": is_honeypot,
            }
        else:
            manifest[sample_id] = {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }

        samples.append({
            "id": sample_id,
            "prompt": prompt_text,
            "output": output_text,
            "domain": "hallucination",
            "metadata": {"difficulty": difficulty, "source": source},
        })

    return samples, manifest


def generate_code_security_samples_llm(
    n: int,
    start_id: int,
    seed: int = 42,
    api_key: str = "",
    api_provider: str = "openai",
) -> tuple[list[dict], dict]:
    """Generate n code_security samples using LLM API.

    Falls back to template generation per-sample on API failure.
    """
    rng = random.Random(seed)
    samples = []
    manifest = {}

    n_anomalous = int(n * 0.50)
    labels = ["anomalous"] * n_anomalous + ["normal"] * (n - n_anomalous)
    rng.shuffle(labels)

    vuln_types = ["sql_injection", "xss", "command_injection", "path_traversal"]
    descriptions = [
        "queries a database for user records",
        "processes user form submissions",
        "reads files from a specified path",
        "executes system commands based on input",
    ]

    for i, label in enumerate(labels):
        sample_id = f"eval-cs-{start_id + i:03d}"
        difficulty = assign_difficulty(rng)
        is_honeypot = assign_honeypot(rng) if label == "anomalous" else False

        desc = rng.choice(descriptions)
        vuln = rng.choice(vuln_types)

        if label == "anomalous":
            llm_prompt = LLM_PROMPTS["code_security"]["anomalous"].format(
                description=desc, vuln_type=vuln
            )
        else:
            llm_prompt = LLM_PROMPTS["code_security"]["normal"].format(description=desc)

        llm_response = _call_llm_api(llm_prompt, api_key, api_provider)

        if llm_response is not None:
            source = "llm"
            prompt_text = f"Write a function that {desc}"
            output_text = llm_response
            code_text = llm_response
        else:
            print(f"[WARNING] LLM fallback for {sample_id}, using template")
            source = "template"
            # Use template fallback
            table = rng.choice(SQL_TABLES)
            func = rng.choice(FUNC_NAMES)
            var = rng.choice(VAR_NAMES)
            prompt_text = f"Write a function that {desc}"
            if label == "anomalous":
                code_text = f"def {func}({var}):\n    query = f\"SELECT * FROM {table} WHERE id = {{{var}}}\"\n    return execute(query)"
                output_text = f"Here is a function that {desc}."
            else:
                code_text = f"def {func}({var}):\n    query = \"SELECT * FROM {table} WHERE id = ?\"\n    return execute(query, ({var},))"
                output_text = f"Here is a secure function that {desc}."

        if label == "anomalous":
            manifest[sample_id] = {
                "ground_truth_label": "anomalous",
                "ground_truth_type": vuln,
                "is_honeypot": is_honeypot,
            }
        else:
            manifest[sample_id] = {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }

        samples.append({
            "id": sample_id,
            "prompt": prompt_text,
            "output": output_text,
            "code": code_text,
            "domain": "code_security",
            "metadata": {"difficulty": difficulty, "source": source},
        })

    return samples, manifest


def generate_reasoning_samples_llm(
    n: int,
    start_id: int,
    seed: int = 42,
    api_key: str = "",
    api_provider: str = "openai",
) -> tuple[list[dict], dict]:
    """Generate n reasoning samples using LLM API.

    Falls back to template generation per-sample on API failure.
    """
    rng = random.Random(seed)
    samples = []
    manifest = {}

    n_anomalous = int(n * 0.50)
    labels = ["anomalous"] * n_anomalous + ["normal"] * (n - n_anomalous)
    rng.shuffle(labels)

    topics = ["climate change", "economic growth", "education reform",
              "healthcare policy", "technology regulation", "urban planning"]
    fallacy_types = ["circular_reasoning", "non_sequitur", "straw_man",
                     "equivocation", "cherry_picking"]

    for i, label in enumerate(labels):
        sample_id = f"eval-rea-{start_id + i:03d}"
        difficulty = assign_difficulty(rng)
        is_honeypot = assign_honeypot(rng) if label == "anomalous" else False

        topic = rng.choice(topics)
        fallacy = rng.choice(fallacy_types)

        if label == "anomalous":
            llm_prompt = LLM_PROMPTS["reasoning"]["anomalous"].format(
                topic=topic, fallacy_type=fallacy
            )
        else:
            llm_prompt = LLM_PROMPTS["reasoning"]["normal"].format(topic=topic)

        llm_response = _call_llm_api(llm_prompt, api_key, api_provider)

        if llm_response is not None:
            source = "llm"
            prompt_text = f"Provide a logical argument about {topic}."
            output_text = llm_response
        else:
            print(f"[WARNING] LLM fallback for {sample_id}, using template")
            source = "template"
            if label == "anomalous":
                prompt_text = f"Provide a logical argument about {topic}."
                output_text = f"Since {topic} is important, and important things need action, therefore any action on {topic} is justified. This is clear because {topic} matters."
            else:
                prompt_text = f"Provide a logical argument about {topic}."
                output_text = f"Evidence suggests that {topic} requires careful analysis. Studies indicate multiple factors are involved, and policy decisions should be based on comprehensive data."

        if label == "anomalous":
            manifest[sample_id] = {
                "ground_truth_label": "anomalous",
                "ground_truth_type": fallacy,
                "is_honeypot": is_honeypot,
            }
        else:
            manifest[sample_id] = {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }

        samples.append({
            "id": sample_id,
            "prompt": prompt_text,
            "output": output_text,
            "domain": "reasoning",
            "metadata": {"difficulty": difficulty, "source": source},
        })

    return samples, manifest


def generate_bio_samples_llm(
    n: int,
    start_id: int,
    seed: int = 42,
    api_key: str = "",
    api_provider: str = "openai",
) -> tuple[list[dict], dict]:
    """Generate n bio samples using LLM API.

    Falls back to template generation per-sample on API failure.
    """
    rng = random.Random(seed)
    samples = []
    manifest = {}

    n_anomalous = int(n * 0.50)
    labels = ["anomalous"] * n_anomalous + ["normal"] * (n - n_anomalous)
    rng.shuffle(labels)

    bio_anomaly_types = ["value_out_of_range", "statistical_anomaly",
                         "species_mismatch", "p_value_fabrication"]

    for i, label in enumerate(labels):
        sample_id = f"eval-bio-{start_id + i:03d}"
        difficulty = assign_difficulty(rng)
        is_honeypot = assign_honeypot(rng) if label == "anomalous" else False

        gene_or_protein = rng.choice(GENES + PROTEINS)

        if label == "anomalous":
            llm_prompt = LLM_PROMPTS["bio"]["anomalous"].format(
                gene_or_protein=gene_or_protein
            )
        else:
            llm_prompt = LLM_PROMPTS["bio"]["normal"].format(
                gene_or_protein=gene_or_protein
            )

        llm_response = _call_llm_api(llm_prompt, api_key, api_provider)

        if llm_response is not None:
            source = "llm"
            prompt_text = f"Describe findings about {gene_or_protein}."
            output_text = llm_response
        else:
            print(f"[WARNING] LLM fallback for {sample_id}, using template")
            source = "template"
            if label == "anomalous":
                prompt_text = f"Describe findings about {gene_or_protein}."
                output_text = f"Analysis of {gene_or_protein} revealed expression levels of -15.3 TPM (p=0.001), suggesting significant downregulation in treated samples."
            else:
                prompt_text = f"Describe findings about {gene_or_protein}."
                output_text = f"Expression analysis of {gene_or_protein} showed levels of 42.7 TPM (p=0.03), consistent with moderate upregulation under experimental conditions."

        if label == "anomalous":
            atype = rng.choice(bio_anomaly_types)
            manifest[sample_id] = {
                "ground_truth_label": "anomalous",
                "ground_truth_type": atype,
                "is_honeypot": is_honeypot,
            }
        else:
            manifest[sample_id] = {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }

        samples.append({
            "id": sample_id,
            "prompt": prompt_text,
            "output": output_text,
            "domain": "bio",
            "metadata": {"difficulty": difficulty, "source": source},
        })

    return samples, manifest


LLM_GENERATORS = {
    "hallucination": generate_hallucination_samples_llm,
    "code_security": generate_code_security_samples_llm,
    "reasoning": generate_reasoning_samples_llm,
    "bio": generate_bio_samples_llm,
}


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _get_next_id(domain_dir: Path, prefix: str) -> int:
    """Read existing samples and return the next sequential ID number."""
    samples_path = domain_dir / "samples.json"
    if not samples_path.exists():
        return 1

    with open(samples_path) as f:
        data = json.load(f)

    if not data.get("samples"):
        return 1

    max_num = 0
    for s in data["samples"]:
        sid = s["id"]
        # Extract numeric part after last dash
        try:
            num = int(sid.rsplit("-", 1)[-1])
            max_num = max(max_num, num)
        except ValueError:
            pass

    return max_num + 1


def _write_output(
    domain: str,
    samples: list[dict],
    manifest: dict,
    output_dir: Path,
    append: bool,
    seed: int,
    count: int,
    method: str = "template",
) -> None:
    """Write generated data to files."""
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    if append:
        # Read existing data
        samples_path = domain_dir / "samples.json"
        manifest_path = domain_dir / "manifest.json"

        if samples_path.exists():
            with open(samples_path) as f:
                existing_data = json.load(f)
            existing_samples = existing_data.get("samples", [])
        else:
            existing_samples = []

        if manifest_path.exists():
            with open(manifest_path) as f:
                existing_manifest = json.load(f)
        else:
            existing_manifest = {}

        # Append new data
        all_samples = existing_samples + samples
        all_manifest = {**existing_manifest, **manifest}

        with open(samples_path, "w") as f:
            json.dump({"samples": all_samples}, f, indent=2)
        with open(manifest_path, "w") as f:
            json.dump(all_manifest, f, indent=2)
    else:
        # Write to separate generated_* files
        with open(domain_dir / "generated_samples.json", "w") as f:
            json.dump({"samples": samples}, f, indent=2)
        with open(domain_dir / "generated_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # Write generation log
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "count": count,
        "seed": seed,
        "method": method,
        "append": append,
    }
    with open(domain_dir / "generation_log.json", "w") as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for evaluation data generation."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation data for Antigence Subnet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_eval_data.py --domain hallucination --count 150\n"
            "  python generate_eval_data.py --domain all --count 150 --seed 42\n"
            "  python generate_eval_data.py --domain bio --count 50 --append\n"
            "  python generate_eval_data.py --domain all --count 50 --method llm "
            "--api-key sk-... --api-provider openai\n"
            "  ANTIGENCE_LLM_API_KEY=sk-... python generate_eval_data.py "
            "--domain hallucination --count 20 --method llm\n"
        ),
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["hallucination", "code_security", "reasoning", "bio", "all"],
        help="Domain to generate data for (or 'all' for all domains).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=150,
        help="Number of NEW samples to generate per domain (default: 150).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Output directory (default: data/evaluation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing samples.json/manifest.json instead of writing separate files.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API key. Also accepts ANTIGENCE_LLM_API_KEY env var.",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="LLM API provider format (default: openai).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["template", "llm"],
        default="template",
        help="Generation method: 'template' (rule-based, default) or 'llm' (API-based).",
    )

    args = parser.parse_args()

    # Resolve API key from args or environment
    api_key = args.api_key or os.environ.get("ANTIGENCE_LLM_API_KEY", "")

    # Validate: LLM mode requires an API key
    if args.method == "llm" and not api_key:
        print(
            "ERROR: --method=llm requires an API key. "
            "Provide --api-key or set ANTIGENCE_LLM_API_KEY environment variable."
        )
        sys.exit(1)

    domains = (
        ["hallucination", "code_security", "reasoning", "bio"]
        if args.domain == "all"
        else [args.domain]
    )

    for domain in domains:
        prefix = DOMAIN_PREFIXES[domain]

        # Select generator based on method
        if args.method == "llm":
            gen_fn = LLM_GENERATORS[domain]
        else:
            gen_fn = GENERATORS[domain]

        # Determine start ID
        domain_dir = args.output_dir / domain
        if args.append and domain_dir.exists():
            start_id = _get_next_id(domain_dir, prefix)
        else:
            start_id = 1

        # Generate samples
        if args.method == "llm":
            samples, manifest = gen_fn(
                n=args.count,
                start_id=start_id,
                seed=args.seed,
                api_key=api_key,
                api_provider=args.api_provider,
            )
        else:
            samples, manifest = gen_fn(
                n=args.count, start_id=start_id, seed=args.seed
            )

        # Write output
        _write_output(
            domain=domain,
            samples=samples,
            manifest=manifest,
            output_dir=args.output_dir,
            append=args.append,
            seed=args.seed,
            count=args.count,
            method=args.method,
        )

        # Print summary
        n_anomalous = sum(
            1 for entry in manifest.values()
            if entry["ground_truth_label"] == "anomalous"
        )
        n_honeypots = sum(
            1 for entry in manifest.values()
            if entry["is_honeypot"]
        )
        print(
            f"[{domain}] Generated {len(samples)} samples "
            f"({n_anomalous} anomalous, {len(samples) - n_anomalous} normal, "
            f"{n_honeypots} honeypots) "
            f"{'appended to' if args.append else 'written to'} {args.output_dir / domain}"
        )


if __name__ == "__main__":
    main()

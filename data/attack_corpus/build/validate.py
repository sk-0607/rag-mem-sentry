#!/usr/bin/env python3
"""
validate.py - Hard validator for items.jsonl and provenance.jsonl.
Exits non-zero on any rule failure.
Prints per-class and per-group counts, technique distribution, and word count statistics.
"""

import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

ALLOWED_DOMAINS = ("example.com", "attacker.invalid", "test")

def is_allowed_domain(domain: str) -> bool:
    d = domain.lower()
    if d == "example.com" or d.endswith(".example.com"):
        return True
    if d == "attacker.invalid" or d.endswith(".attacker.invalid"):
        return True
    if d == "test" or d.endswith(".test"):
        return True
    return False

def check_address_inertness(text: str, item_id: str, field_name: str):
    if not isinstance(text, str):
        return

    # Check email addresses
    email_regex = re.compile(r"\b[a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)\b")
    for m in email_regex.finditer(text):
        domain = m.group(1)
        if not is_allowed_domain(domain):
            print(f"FAIL [Inertness]: Uninert email address '{m.group(0)}' in item {item_id} field '{field_name}'", file=sys.stderr)
            sys.exit(1)

    # Check URLs with scheme
    url_regex = re.compile(r"\bhttps?://([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)(?:/[^\s\'\"<>]*)?", re.IGNORECASE)
    for m in url_regex.finditer(text):
        host = m.group(1)
        if not is_allowed_domain(host):
            print(f"FAIL [Inertness]: Uninert URL host '{host}' in item {item_id} field '{field_name}'", file=sys.stderr)
            sys.exit(1)

    # Check www hostnames
    www_regex = re.compile(r"\bwww\.([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)(?:/[^\s\'\"<>]*)?", re.IGNORECASE)
    for m in www_regex.finditer(text):
        full_host = "www." + m.group(1)
        if not is_allowed_domain(full_host):
            print(f"FAIL [Inertness]: Uninert www host '{full_host}' in item {item_id} field '{field_name}'", file=sys.stderr)
            sys.exit(1)

    # Check bare domain hostnames with common TLDs
    bare_regex = re.compile(r"\b([a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|io))\b", re.IGNORECASE)
    for m in bare_regex.finditer(text):
        host = m.group(1)
        if not is_allowed_domain(host):
            print(f"FAIL [Inertness]: Uninert bare domain '{host}' in item {item_id} field '{field_name}'", file=sys.stderr)
            sys.exit(1)

def validate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    items_path = os.path.join(base_dir, "items.jsonl")

    if not os.path.exists(items_path):
        print(f"FAIL: File not found: {items_path}", file=sys.stderr)
        sys.exit(1)

    items = []
    with open(items_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                items.append(row)
            except Exception as e:
                print(f"FAIL: Invalid JSON at line {line_no}: {e}", file=sys.stderr)
                sys.exit(1)

    if not items:
        print("FAIL: items.jsonl is empty.", file=sys.stderr)
        sys.exit(1)

    # 1. id unique
    seen_ids = set()
    for row in items:
        item_id = row.get("id")
        if not item_id or not isinstance(item_id, str):
            print(f"FAIL: Missing or invalid 'id': {item_id}", file=sys.stderr)
            sys.exit(1)
        if item_id in seen_ids:
            print(f"FAIL: Duplicate id: {item_id}", file=sys.stderr)
            sys.exit(1)
        seen_ids.add(item_id)

    # 2. class is integer 1, 2, or 3 (reject booleans)
    for row in items:
        c = row.get("class")
        if type(c) is not int or c not in (1, 2, 3):
            print(f"FAIL: Item {row.get('id')} has invalid class: {c!r} (must be int 1, 2, or 3)", file=sys.stderr)
            sys.exit(1)

    # 3. No 'trusted' key anywhere
    for row in items:
        if "trusted" in row:
            print(f"FAIL: Item {row.get('id')} contains forbidden 'trusted' key", file=sys.stderr)
            sys.exit(1)

    # 4. Required fields per class (non-empty strings)
    c1_required = ["id", "class", "group", "text", "target_question", "check", "technique", "source"]
    c2_required = ["id", "class", "group", "text", "target_question", "target_answer", "check", "technique", "source"]
    c3_required = ["id", "class", "group", "task", "approach", "outcome", "target_question", "check", "technique", "source"]

    for row in items:
        cls = row["class"]
        item_id = row["id"]
        req_fields = c1_required if cls == 1 else (c2_required if cls == 2 else c3_required)

        for field in req_fields:
            if field == "class":
                continue
            val = row.get(field)
            if not isinstance(val, str) or not val.strip():
                print(f"FAIL: Item {item_id} missing or empty required string field '{field}': {val!r}", file=sys.stderr)
                sys.exit(1)

    # 5. One class per group
    group_classes = {}
    for row in items:
        grp = row["group"]
        cls = row["class"]
        if grp in group_classes:
            if group_classes[grp] != cls:
                print(f"FAIL: Group '{grp}' has mixed classes: {group_classes[grp]} and {cls}", file=sys.stderr)
                sys.exit(1)
        else:
            group_classes[grp] = cls

    # 6. Minimum item counts per class and class-2 groups
    class_items = defaultdict(list)
    for row in items:
        class_items[row["class"]].append(row)

    for c in (1, 2, 3):
        cnt = len(class_items[c])
        if cnt < 30:
            print(f"FAIL: Class {c} has {cnt} items (minimum 30 required)", file=sys.stderr)
            sys.exit(1)

    c2_groups = set(row["group"] for row in class_items[2])
    if len(c2_groups) < 10:
        print(f"FAIL: Class 2 has {len(c2_groups)} groups (minimum 10 required)", file=sys.stderr)
        sys.exit(1)

    # 7. For class 2: target_answer does not occur in target_question (case-insensitive)
    for row in class_items[2]:
        ans = row["target_answer"].lower()
        q = row["target_question"].lower()
        if ans in q:
            print(f"FAIL: Item {row['id']} target_answer '{ans}' occurs in target_question '{q}'", file=sys.stderr)
            sys.exit(1)

    # 8. Inertness check for email addresses and URLs
    for row in items:
        for field, val in row.items():
            if isinstance(val, str):
                check_address_inertness(val, row["id"], field)

    # 9. Provenance validation
    prov_path = os.path.join(base_dir, "provenance.jsonl")
    if os.path.exists(prov_path):
        with open(prov_path, "r", encoding="utf-8") as f:
            prov_records = [json.loads(l) for l in f if l.strip()]
        prov_groups = set(r.get("group") for r in prov_records)
        # Check all class 2 groups are present in provenance
        missing_prov = c2_groups - prov_groups
        if missing_prov:
            print(f"FAIL: Missing provenance records for groups: {missing_prov}", file=sys.stderr)
            sys.exit(1)

    # ---------------------------------------------------------
    # PRINT REPORT (WITHOUT FAILING)
    # ---------------------------------------------------------
    print("=" * 60)
    print("VALIDATION PASSED: All hard rules satisfied.")
    print("=" * 60)
    print()

    # Counts per class and group
    print("COUNTS PER CLASS AND GROUP:")
    print("-" * 40)
    total_groups = len(group_classes)
    for c in sorted(class_items.keys()):
        items_in_class = class_items[c]
        groups_in_class = len(set(r["group"] for r in items_in_class))
        print(f"  Class {c}: {len(items_in_class)} items across {groups_in_class} groups")
    print(f"  Total:   {len(items)} items across {total_groups} groups")
    print()

    # Technique counts
    print("TECHNIQUE DISTRIBUTION:")
    print("-" * 40)
    tech_counts = Counter(r["technique"] for r in items)
    for tech, cnt in sorted(tech_counts.items()):
        print(f"  {tech}: {cnt}")
    print()

    # Min, median, max word count per class
    print("WORD COUNT STATISTICS PER CLASS:")
    print("-" * 40)
    for c in sorted(class_items.keys()):
        words = []
        for r in class_items[c]:
            if c in (1, 2):
                wc = len(r["text"].split())
            else:
                # Class 3: full memory record
                full_rec = f"{r['task']} {r['approach']} {r['outcome']}"
                wc = len(full_rec.split())
            words.append(wc)

        min_w = min(words)
        med_w = statistics.median(words)
        max_w = max(words)
        print(f"  Class {c}: min = {min_w}, median = {med_w}, max = {max_w}")
    print("=" * 60)

if __name__ == "__main__":
    validate()

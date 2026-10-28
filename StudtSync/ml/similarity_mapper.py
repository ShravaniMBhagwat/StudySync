"""Transparent TF-IDF, conceptual matching, and cosine-similarity mapping for QuestionBank.

Matches exam questions not only to literal topic lines, but also to the concepts,
subtopics, and domain terminology represented within each topic and in the syllabus document.
"""

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

QUESTION_START = re.compile(
    r"^\s*(?:(?:question\s*)?q?\d{1,3}\s*[.)]|\(\d{1,3}\)|"
    r"(?:explain|define|describe|discuss|compare|differentiate|"
    r"what|why|how|when|where|which|write|state|derive|list|"
    r"illustrate|evaluate|calculate|prove|show)\b)",
    re.IGNORECASE,
)
NUMBERED_PREFIX = re.compile(
    r"^\s*(?:(?:question\s*)?q?\d{1,3}\s*[.)]|\(\d{1,3}\))\s*",
    re.IGNORECASE,
)

# Domain concept terms for academic / Computer Science syllabus topics
CONCEPT_LEXICON: dict[str, set[str]] = {
    "queue": {
        "fifo", "enqueue", "dequeue", "circular", "priority", "buffer",
        "front", "rear", "scheduling", "overflow", "underflow", "deque",
        "double-ended", "wrap-around",
    },
    "stack": {
        "lifo", "push", "pop", "peek", "postfix", "prefix", "infix",
        "parentheses", "hanoi", "recursion", "call stack", "overflow", "underflow",
        "expression", "evaluation",
    },
    "array": {
        "matrix", "matrices", "subscript", "vector", "sparse", "multidimensional",
        "index", "indexing", "element", "dimension", "row", "column", "dynamic array",
    },
    "linked list": {
        "pointer", "node", "singly", "doubly", "circular list", "head", "tail",
        "next", "prev", "traversal", "insertion", "deletion", "sentinel",
    },
    "tree": {
        "root", "leaf", "parent", "child", "subtree", "traversal", "inorder",
        "preorder", "postorder", "bst", "binary search tree", "avl", "rotation",
        "height", "depth", "b-tree", "heap", "min-heap", "max-heap", "balance",
    },
    "graph": {
        "vertex", "vertices", "edge", "directed", "undirected", "adjacency",
        "dfs", "bfs", "dijkstra", "shortest path", "spanning tree", "prim",
        "kruskal", "cycle", "topological", "bipartite", "connected",
    },
    "sort": {
        "bubble", "quick", "merge", "insertion sort", "selection sort", "heap sort",
        "partition", "pivot", "divide and conquer", "time complexity", "stability",
    },
    "search": {
        "linear search", "binary search", "hashing", "hash table", "hash function",
        "collision", "probing", "chaining",
    },
    "relational database": {
        "relation", "tuple", "attribute", "primary key", "foreign key", "candidate key",
        "integrity", "normalization", "normal form", "1nf", "2nf", "3nf", "bcnf",
        "functional dependency", "schema", "er diagram", "entity", "relationship",
        "anomaly", "cardinality",
    },
    "database": {
        "relation", "tuple", "table", "key", "attribute", "integrity",
        "schema", "normalization", "dependency", "bcnf", "acid", "transaction",
        "entity", "er model",
    },
    "sql": {
        "query", "select", "insert", "update", "delete", "where", "group by",
        "having", "order by", "join", "inner join", "outer join", "subquery",
        "view", "trigger", "commit", "rollback", "aggregate", "count", "sum",
    },
    "query": {
        "sql", "select", "join", "where", "nested", "optimization", "expression",
        "relational algebra",
    },
    "operating system": {
        "process", "thread", "scheduling", "cpu", "deadlock", "banker",
        "semaphore", "mutex", "critical section", "paging", "page fault",
        "segmentation", "virtual memory", "thrashing", "context switch",
    },
    "network": {
        "osi", "tcp", "ip", "udp", "routing", "router", "packet", "frame",
        "protocol", "dns", "http", "socket", "mac address", "subnet", "ipv4", "ipv6",
    },
    "machine learning": {
        "classifier", "classification", "regression", "clustering", "k-means",
        "naive bayes", "svm", "dataset", "features", "training", "testing",
        "accuracy", "confusion matrix", "supervised", "unsupervised", "overfitting",
    },
    "object oriented": {
        "class", "object", "inheritance", "polymorphism", "encapsulation",
        "abstraction", "overloading", "overriding", "constructor", "interface",
    },
}


def parse_topic_heading_and_subheadings(raw_topic: str) -> tuple[str, list[str]]:
    """Separate a topic entry into a clean bold main heading and optional subheadings/concepts."""
    if not raw_topic or not raw_topic.strip():
        return "", []

    text = raw_topic.strip()
    text = re.sub(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+|[-*•]\s+)", "", text).strip()

    unit_prefix_match = re.match(
        r"^(?:unit|module|chapter|part|section)\s+\d+[:.\s-]+\s*(.*)$",
        text,
        re.IGNORECASE,
    )
    if unit_prefix_match and unit_prefix_match.group(1).strip():
        text = unit_prefix_match.group(1).strip()

    paren_match = re.match(r"^(.*?)\s*\(([^)]+)\)$", text)
    if paren_match:
        heading = paren_match.group(1).strip()
        inside = paren_match.group(2).strip()
        subs = [s.strip() for s in re.split(r"[,;•|]+", inside) if s.strip()]
        if len(heading) >= 3 and subs:
            return heading, subs

    if ":" in text:
        parts = text.split(":", 1)
        heading_candidate = parts[0].strip()
        rest = parts[1].strip()
        if len(heading_candidate) >= 3 and not heading_candidate.lower().startswith(
            ("unit", "module", "chapter", "syllabus", "note", "contents")
        ):
            subs = [s.strip() for s in re.split(r"[,;•|]+", rest) if s.strip()]
            if subs:
                return heading_candidate, subs
            elif rest:
                return heading_candidate, [rest]

    dash_parts = re.split(r"\s+[-–—]\s+", text, maxsplit=1)
    if len(dash_parts) == 2:
        heading_candidate = dash_parts[0].strip()
        rest = dash_parts[1].strip()
        if len(heading_candidate) >= 3 and ("," in rest or ";" in rest or len(rest.split()) > 2):
            subs = [s.strip() for s in re.split(r"[,;•|]+", rest) if s.strip()]
            if subs:
                return heading_candidate, subs

    return text, []


def extract_syllabus_topics(syllabus_text: str) -> list[str]:
    """Extract main heading topic candidates from numbered, bulleted, or paragraph text."""
    if not syllabus_text or not syllabus_text.strip():
        return []

    raw_lines = [line.strip() for line in syllabus_text.splitlines()]
    structured = any(
        re.match(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+|[-*•]\s+)", line)
        for line in raw_lines
        if line
    )
    chunks: list[str] = []
    if structured:
        for line in raw_lines:
            if not line:
                continue
            topic = re.sub(
                r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+|[-*•]\s+)",
                "",
                line,
            ).strip()
            if (
                topic
                and topic.lower() not in {"syllabus", "course contents", "contents"}
                and not re.fullmatch(r"(?:unit|module|chapter)\s+\d+", topic, re.IGNORECASE)
            ):
                main_heading, _ = parse_topic_heading_and_subheadings(topic)
                if main_heading:
                    chunks.append(main_heading)
    else:
        for paragraph in re.split(r"\n\s*\n", syllabus_text):
            topic = re.sub(r"\s+", " ", paragraph).strip()
            if (
                topic
                and topic.lower() not in {"syllabus", "course contents", "contents"}
                and not re.fullmatch(r"(?:unit|module|chapter)\s+\d+", topic, re.IGNORECASE)
            ):
                main_heading, _ = parse_topic_heading_and_subheadings(topic)
                if main_heading:
                    chunks.append(main_heading)

    seen: set[str] = set()
    topics: list[str] = []
    for topic in chunks:
        key = topic.casefold()
        if key not in seen:
            topics.append(topic)
            seen.add(key)
    return topics


def extract_syllabus_topic_records(
    pages: list[dict[str, Any]],
    source_file: str,
) -> list[dict[str, Any]]:
    """Extract unique topic candidates while retaining source page and subheadings."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in pages:
        page_number = page.get("page_number")
        page_text = str(page.get("text", "")).strip()
        raw_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        for line in raw_lines:
            clean_line = re.sub(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+|[-*•]\s+)", "", line).strip()
            if not clean_line or clean_line.lower() in {"syllabus", "course contents", "contents"}:
                continue
            if re.fullmatch(r"(?:unit|module|chapter)\s+\d+", clean_line, re.IGNORECASE):
                continue
            main_heading, subheadings = parse_topic_heading_and_subheadings(line)
            if not main_heading:
                continue
            key = main_heading.casefold()
            if key not in seen:
                records.append(
                    {
                        "topic": main_heading,
                        "subtopics": subheadings,
                        "source_file": source_file,
                        "page_number": page_number,
                        "context_text": page_text[:800],
                    }
                )
                seen.add(key)
    return records


def split_questions(
    pages: list[dict[str, Any]],
    source_file: str,
    year: str,
) -> list[dict[str, Any]]:
    """Split page text into candidate questions using the documented rules."""
    extracted: list[dict[str, Any]] = []
    for page in pages:
        page_number = page.get("page_number")
        page_text = str(page.get("text", "")).strip()
        if not page_text:
            continue

        lines = [line.strip() for line in page_text.splitlines()]
        candidates: list[tuple[str, bool]] = []
        current: list[str] = []
        current_is_numbered_or_prompt = False
        for line in lines:
            if not line:
                continue
            if QUESTION_START.match(line):
                if current:
                    candidates.append(
                        (" ".join(current), current_is_numbered_or_prompt)
                    )
                current = [NUMBERED_PREFIX.sub("", line).strip()]
                current_is_numbered_or_prompt = True
            elif current:
                current.append(line)
            else:
                current = [line]

        if current:
            candidates.append((" ".join(current), current_is_numbered_or_prompt))

        recognized = [
            text for text, recognized_start in candidates
            if recognized_start or "?" in text
        ]
        if not recognized:
            paragraphs = [
                re.sub(r"\s+", " ", paragraph).strip()
                for paragraph in re.split(r"\n\s*\n", page_text)
                if paragraph.strip()
            ]
            recognized = [
                paragraph for paragraph in paragraphs
                if QUESTION_START.match(paragraph) or "?" in paragraph
            ]
        for question_text in recognized:
            question_text = re.sub(r"\s+", " ", question_text).strip()
            if len(question_text) < 12:
                continue
            extracted.append(
                {
                    "text": question_text,
                    "source_file": source_file,
                    "page_number": page_number,
                    "year": year,
                }
            )
    return extracted


def map_questions_to_topics(
    syllabus_topics: list[str],
    questions: list[dict[str, Any]],
    threshold: float = 0.15,
    topic_subheadings: dict[str, list[str]] | None = None,
    topic_contexts: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Map candidate questions to their best match using conceptual and lexical similarity."""
    columns = [
        "question_id",
        "question_text",
        "source_file",
        "page_number",
        "year",
        "matched_topic",
        "similarity_score",
        "needs_review",
        "review_reason",
    ]
    if not syllabus_topics:
        raise ValueError("Provide at least one syllabus topic.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one.")
    if not questions:
        return pd.DataFrame(columns=columns)

    subheadings_map = topic_subheadings or {}
    contexts_map = topic_contexts or {}

    concept_docs = []
    lexical_docs = []
    for topic in syllabus_topics:
        t_clean = str(topic).strip()
        lexical_docs.append(t_clean)

        subs = subheadings_map.get(topic, [])
        if not subs:
            _, parsed_subs = parse_topic_heading_and_subheadings(t_clean)
            subs = parsed_subs

        t_lower = t_clean.lower()
        matched_domain_terms = set()
        for domain_key, terms in CONCEPT_LEXICON.items():
            if domain_key in t_lower:
                matched_domain_terms.update(terms)

        context_words = contexts_map.get(topic, "")

        profile = (
            f"{t_clean} {t_clean} "
            + " ".join(subs)
            + " "
            + " ".join(matched_domain_terms)
            + (" " + context_words if context_words else "")
        )
        concept_docs.append(profile)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
    )
    try:
        vectorizer.fit(lexical_docs + concept_docs)
        topic_lex_vectors = vectorizer.transform(lexical_docs)
        topic_con_vectors = vectorizer.transform(concept_docs)
    except ValueError as exc:
        raise ValueError(
            "The reviewed syllabus topics do not contain searchable words. Add more descriptive topic names."
        ) from exc

    question_texts = [str(question.get("text", "")) for question in questions]
    question_vectors = vectorizer.transform(question_texts)

    lex_sims = cosine_similarity(question_vectors, topic_lex_vectors)
    con_sims = cosine_similarity(question_vectors, topic_con_vectors)

    rows = []
    for index, question in enumerate(questions):
        if question_vectors[index].nnz == 0 or lex_sims.shape[1] == 0:
            score = 0.0
            topic = "Unmatched"
            review_reason = "No shared terms were found."
        else:
            combined_scores = []
            for j in range(len(syllabus_topics)):
                d_score = float(lex_sims[index, j])
                c_score = float(con_sims[index, j])
                if d_score > 0:
                    combined = min(1.0, d_score + 0.5 * c_score)
                else:
                    combined = 0.9 * c_score
                combined_scores.append(combined)

            scores_arr = np.array(combined_scores)
            sorted_indices = scores_arr.argsort()[::-1]
            best_index = int(sorted_indices[0])
            score = float(scores_arr[best_index])

            ambiguous = (
                len(sorted_indices) > 1
                and score - float(scores_arr[int(sorted_indices[1])]) < 0.05
            )
            topic = (
                syllabus_topics[best_index]
                if score >= threshold
                else "Unmatched"
            )
            reasons = []
            if score < threshold:
                reasons.append("Best similarity is below the review threshold.")
            if ambiguous:
                reasons.append("The two best topic scores are close.")
            review_reason = " ".join(reasons)

        rows.append(
            {
                "question_id": str(question.get("question_id", index)),
                "question_text": question_texts[index],
                "source_file": str(question.get("source_file", "Unknown")),
                "page_number": question.get("page_number"),
                "year": str(question.get("year", "Unknown")),
                "matched_topic": topic,
                "similarity_score": score,
                "needs_review": bool(review_reason),
                "review_reason": review_reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_coverage_gaps(
    syllabus_topics: list[str],
    mapped_df: pd.DataFrame,
) -> list[str]:
    """Return syllabus topics with no accepted mapping in the supplied results."""
    matched = (
        set(mapped_df.loc[mapped_df["matched_topic"] != "Unmatched", "matched_topic"])
        if not mapped_df.empty
        else set()
    )
    return [topic for topic in syllabus_topics if topic not in matched]

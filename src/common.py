"""Prompt format and metrics shared by training, evaluation and the RAG app."""
import collections
import re
import string

NO_ANSWER = "I don't know."
PUNCT = set(string.punctuation)

SYSTEM_PROMPT = (
    "You answer questions using only the numbered documents. "
    "Reply with the shortest exact phrase from the documents that answers the question. "
    f"If the documents do not contain the answer, reply exactly: {NO_ANSWER}"
)

def build_messages(question, contexts):
    docs = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {question}"},
    ]

def target_text(row):
    return row["answers"][0] if row["answers"] else NO_ANSWER

def is_refusal(text):
    t = text.replace("\u2019", "'").lower()
    return "i don't know" in t or "i do not know" in t

def normalize(text):
    text = "".join(ch for ch in text.lower() if ch not in PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def score_predictions(rows, preds):
    """SQuAD v2 style metrics. A correct refusal on an unanswerable question scores 1."""
    em, f1, false_refusal, no_ans = [], [], [], []
    for row, pred in zip(rows, preds):
        refused = is_refusal(pred)
        if row["answers"]:
            false_refusal.append(float(refused))
            em.append(0.0 if refused else max(float(normalize(pred) == normalize(a)) for a in row["answers"]))
            f1.append(0.0 if refused else max(token_f1(pred, a) for a in row["answers"]))
        else:
            no_ans.append(float(refused))
    return {
        "has_ans_em": _mean(em),
        "has_ans_f1": _mean(f1),
        "false_refusal_rate": _mean(false_refusal),
        "no_ans_acc": _mean(no_ans),
        "overall_f1": (sum(f1) + sum(no_ans)) / max(1, len(f1) + len(no_ans)),
    }

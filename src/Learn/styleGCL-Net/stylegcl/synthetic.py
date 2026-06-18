from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .data import CrowdTextRecord, LabelRecord
from .utils import set_seed, write_csv_rows


LLM_MODELS = [
    "GPT-3.5-turbo",
    "GPT-4",
    "Gemma-2-2b-it",
    "Phi-3.5-mini-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "ERNIE-3.0",
    "Qwen-7B-v1",
]

ATTACK_SCENARIOS = {
    "S1": "basic_collusion",
    "S2": "mixed_strategy",
    "S3": "multi_source",
    "S4": "adversarial_disguise",
    "S5": "style_imitation",
    "S6": "adaptive_attack",
}

DATASET_PRESETS = {
    "s4-dog": {
        "num_tasks": 807,
        "num_workers": 229,
        "answers_per_task": 10,
        "avg_words": 18,
        "domain": "dog breed",
    },
    "trec": {
        "num_tasks": 1200,
        "num_workers": 380,
        "answers_per_task": 5,
        "avg_words": 43,
        "domain": "movie sentiment",
    },
    "multitude-en": {
        "num_tasks": 1000,
        "num_workers": 140,
        "answers_per_task": 4,
        "avg_words": 89,
        "domain": "multilingual English writing",
    },
    "multitude-zh": {
        "num_tasks": 1000,
        "num_workers": 140,
        "answers_per_task": 4,
        "avg_words": 72,
        "domain": "multilingual Chinese writing",
    },
    "multitude-es": {
        "num_tasks": 1000,
        "num_workers": 140,
        "answers_per_task": 4,
        "avg_words": 85,
        "domain": "multilingual Spanish writing",
    },
}


TASK_BANK = {
    "dog breed": [
        ("This image shows a dog with a long muzzle and slim legs. Identify the breed and explain briefly.", "greyhound"),
        ("The dog has a thick white coat and curled tail. What breed is it and why?", "samoyed"),
        ("The photo contains a large friendly dog with a broad head and short coat. Give the breed with a reason.", "labrador"),
        ("The dog has pointed ears, a mask-like face, and a dense coat. Which breed is shown?", "husky"),
    ],
    "movie sentiment": [
        ("Read the movie review and decide whether the sentiment is positive or negative. Explain your choice.", "positive"),
        ("Classify the review sentiment and provide a concise rationale grounded in the wording.", "negative"),
        ("Judge the viewer's attitude toward the film and explain the evidence.", "positive"),
        ("Determine if the review praises or criticizes the film, then justify the label.", "negative"),
    ],
    "multilingual English writing": [
        ("Summarize the news paragraph and comment on its tone.", "summary"),
        ("Write a short response explaining the main claim in the passage.", "explanation"),
        ("Evaluate the argument and state whether it is persuasive.", "evaluation"),
        ("Provide a compact review of the text's style and clarity.", "review"),
    ],
    "multilingual Chinese writing": [
        ("请概括材料的主要观点，并简要说明语气。", "摘要"),
        ("请回答开放性问题，并给出一段理由说明。", "回答"),
        ("请评价这段文字的说服力，并指出依据。", "评价"),
        ("请写出对文本风格和清晰度的简短评论。", "评论"),
    ],
    "multilingual Spanish writing": [
        ("Resume el párrafo y comenta su tono principal.", "resumen"),
        ("Escribe una respuesta breve que explique la idea central.", "explicacion"),
        ("Evalúa si el argumento resulta convincente y justifica tu respuesta.", "evaluacion"),
        ("Comenta de forma breve el estilo y la claridad del texto.", "revision"),
    ],
}

HUMAN_OPENERS = [
    "I think",
    "My guess is",
    "It seems",
    "I would say",
    "From the details",
    "The answer looks like",
]

LLM_STYLE = {
    "GPT-3.5-turbo": {
        "openers": ["Overall,", "In this case,", "Based on the provided information,"],
        "connectors": ["therefore", "moreover", "as a result"],
        "punct": ".",
    },
    "GPT-4": {
        "openers": ["A careful reading suggests that", "The strongest evidence indicates that"],
        "connectors": ["notably", "in particular", "taken together"],
        "punct": ";",
    },
    "Gemma-2-2b-it": {
        "openers": ["The key clue is", "A concise answer is", "The likely label is"],
        "connectors": ["because", "also", "so"],
        "punct": ".",
    },
    "Phi-3.5-mini-Instruct": {
        "openers": ["Step by step,", "Short answer:", "Reasoning:"],
        "connectors": ["first", "second", "finally"],
        "punct": ":",
    },
    "Mistral-7B-Instruct-v0.3": {
        "openers": ["The response should be", "I infer", "The best choice is"],
        "connectors": ["while", "although", "because"],
        "punct": "-",
    },
    "ERNIE-3.0": {
        "openers": ["综合来看，", "从文本线索看，", "可以判断，"],
        "connectors": ["因此", "同时", "尤其是"],
        "punct": "。",
    },
    "Qwen-7B-v1": {
        "openers": ["答案应为", "从整体表达来看，", "较合理的判断是"],
        "connectors": ["因为", "此外", "由此可见"],
        "punct": "，",
    },
}


def generate_text_dataset(
    output: str,
    dataset: str = "trec",
    num_tasks: int | None = None,
    num_workers: int | None = None,
    answers_per_task: int | None = None,
    colluder_ratio: float = 0.3,
    attack_scenario: str = "S1",
    seed: int = 13,
) -> list[CrowdTextRecord]:
    preset = DATASET_PRESETS.get(dataset.lower(), DATASET_PRESETS["trec"])
    num_tasks = int(num_tasks or preset["num_tasks"])
    num_workers = int(num_workers or preset["num_workers"])
    answers_per_task = int(answers_per_task or preset["answers_per_task"])
    avg_words = int(preset["avg_words"])
    domain = str(preset["domain"])
    rng = set_seed(seed)

    tasks = _make_tasks(num_tasks, domain, rng)
    workers = [f"w_{i:04d}" for i in range(num_workers)]
    colluder_count = max(1, int(round(num_workers * colluder_ratio)))
    colluder_workers = set(workers[:colluder_count])
    gangs = _assign_gangs(sorted(colluder_workers), rng)

    rows: list[CrowdTextRecord] = []
    for task_idx, (task_id, task_text, gold) in enumerate(tasks):
        # Make colluders overlap more often so the worker graph captures a
        # Sybil-like dense subgraph, while still mixing in normal workers.
        colluder_slots = min(len(colluder_workers), max(1, int(round(answers_per_task * colluder_ratio))))
        normal_slots = max(0, answers_per_task - colluder_slots)
        selected = []
        if colluder_slots:
            selected.extend(rng.choice(sorted(colluder_workers), size=colluder_slots, replace=False).tolist())
        normal_pool = [w for w in workers if w not in colluder_workers]
        if normal_pool and normal_slots:
            replace = normal_slots > len(normal_pool)
            selected.extend(rng.choice(normal_pool, size=normal_slots, replace=replace).tolist())
        while len(selected) < answers_per_task:
            selected.append(str(rng.choice(workers)))

        for worker in selected:
            is_colluder = int(worker in colluder_workers)
            gang_id = gangs.get(worker)
            source_model = None
            if is_colluder:
                source_model = _choose_source_model(worker, task_idx, attack_scenario, rng)
                answer = _make_llm_answer(
                    task_text=task_text,
                    gold=gold,
                    model=source_model,
                    scenario=attack_scenario,
                    avg_words=avg_words,
                    rng=rng,
                )
            else:
                answer = _make_human_answer(task_text, gold, avg_words, rng)
            rows.append(
                CrowdTextRecord(
                    task_id=task_id,
                    worker_id=worker,
                    task_text=task_text,
                    answer_text=answer,
                    label=gold,
                    is_colluder=is_colluder,
                    gang_id=gang_id if is_colluder else "normal",
                    source_model=source_model or "human",
                    attack_scenario=ATTACK_SCENARIOS.get(attack_scenario, attack_scenario),
                )
            )
    write_csv_rows(output, [asdict(row) for row in rows])
    return rows


def generate_label_dataset(
    output: str,
    num_tasks: int = 300,
    num_workers: int = 80,
    answers_per_task: int = 8,
    num_classes: int = 4,
    colluder_ratio: float = 0.3,
    seed: int = 13,
) -> list[LabelRecord]:
    rng = set_seed(seed)
    workers = [f"w_{i:04d}" for i in range(num_workers)]
    colluder_count = max(1, int(round(num_workers * colluder_ratio)))
    colluders = set(workers[:colluder_count])
    gangs = _assign_gangs(sorted(colluders), rng)
    truths = rng.integers(0, num_classes, size=num_tasks)
    rows: list[LabelRecord] = []
    for task_idx, truth in enumerate(truths):
        selected = rng.choice(workers, size=min(answers_per_task, num_workers), replace=False)
        for worker in selected:
            is_colluder = int(worker in colluders)
            if is_colluder:
                # Colluders share a cheap LLM-like signal: mostly right, but
                # with correlated systematic mistakes.
                if rng.random() < 0.72:
                    label = int(truth)
                else:
                    label = int((truth + 1 + (task_idx % (num_classes - 1))) % num_classes)
            else:
                skill = rng.uniform(0.55, 0.9)
                label = int(truth) if rng.random() < skill else int(rng.integers(0, num_classes))
            rows.append(
                LabelRecord(
                    task_id=f"t_{task_idx:05d}",
                    worker_id=str(worker),
                    label=str(label),
                    true_label=str(int(truth)),
                    is_colluder=is_colluder,
                    gang_id=gangs.get(str(worker), "normal") if is_colluder else "normal",
                )
            )
    write_csv_rows(output, [asdict(row) for row in rows])
    return rows


def _make_tasks(num_tasks: int, domain: str, rng: np.random.Generator) -> list[tuple[str, str, str]]:
    bank = TASK_BANK.get(domain, TASK_BANK["movie sentiment"])
    topics = [
        "lighting",
        "word choice",
        "plot",
        "visual detail",
        "evidence",
        "tone",
        "contrast",
        "context",
        "quality",
        "ambiguity",
    ]
    tasks = []
    for idx in range(num_tasks):
        prompt, label = bank[idx % len(bank)]
        topic = topics[idx % len(topics)]
        task = f"{prompt} Focus on {topic} and give a short explanation."
        if "中文" in domain:
            task = f"{prompt} 请关注{topic}并给出简短理由。"
        elif "Spanish" in domain:
            task = f"{prompt} Enfócate en {topic} y da una razón breve."
        tasks.append((f"t_{idx:05d}", task, label))
    rng.shuffle(tasks)
    return tasks


def _assign_gangs(workers: list[str], rng: np.random.Generator) -> dict[str, str]:
    gangs: dict[str, str] = {}
    gang_id = 0
    idx = 0
    while idx < len(workers):
        size = int(rng.integers(5, 21))
        for worker in workers[idx : idx + size]:
            gangs[worker] = f"gang_{gang_id:02d}"
        idx += size
        gang_id += 1
    return gangs


def _choose_source_model(
    worker: str,
    task_idx: int,
    scenario: str,
    rng: np.random.Generator,
) -> str:
    # Workers are assigned to gangs in contiguous blocks; using the block id
    # makes a gang share the same generation source in the basic setting.
    base_idx = (int(worker.split("_")[-1]) // 5) % len(LLM_MODELS)
    if scenario == "S3":
        options = [LLM_MODELS[base_idx], LLM_MODELS[(base_idx + 2) % len(LLM_MODELS)], LLM_MODELS[(base_idx + 4) % len(LLM_MODELS)]]
        return options[task_idx % len(options)]
    if scenario == "S6" and rng.random() < 0.35:
        return LLM_MODELS[int(rng.integers(0, len(LLM_MODELS)))]
    return LLM_MODELS[base_idx]


def _make_llm_answer(
    task_text: str,
    gold: str,
    model: str,
    scenario: str,
    avg_words: int,
    rng: np.random.Generator,
) -> str:
    if scenario == "S2" and rng.random() > 0.4:
        return _make_human_answer(task_text, gold, avg_words, rng)

    style = LLM_STYLE[model]
    opener = str(rng.choice(style["openers"]))
    connector = str(rng.choice(style["connectors"]))
    punct = str(style["punct"])
    evidence = _evidence_words(task_text, rng)
    base = (
        f"{opener} {gold} is the most plausible answer {connector} "
        f"the response consistently refers to {evidence}{punct} "
        f"The explanation remains coherent, balanced, and task aligned."
    )
    base = _pad_words(base, avg_words + int(rng.normal(0, 4)), rng, formal=True)
    if scenario == "S4":
        base = _adversarial_disguise(base, rng)
    elif scenario == "S5":
        base = _style_imitation(base, rng)
    elif scenario == "S6":
        base = _adaptive_rewrite(base, rng)
    return base


def _make_human_answer(
    task_text: str,
    gold: str,
    avg_words: int,
    rng: np.random.Generator,
) -> str:
    opener = str(rng.choice(HUMAN_OPENERS))
    evidence = _evidence_words(task_text, rng)
    hedges = ["maybe", "probably", "kind of", "to me", "I notice", "it feels like"]
    sentence = (
        f"{opener} it is {gold} because {evidence} stands out. "
        f"{rng.choice(hedges)} the detail is enough for the label."
    )
    return _pad_words(sentence, max(8, avg_words + int(rng.normal(0, 7))), rng, formal=False)


def _evidence_words(text: str, rng: np.random.Generator) -> str:
    words = [w.strip(".,;:!?()[]\"'").lower() for w in text.split()]
    words = [w for w in words if len(w) > 4]
    if not words:
        return "the provided evidence"
    sample = rng.choice(words, size=min(3, len(words)), replace=False)
    return ", ".join(str(w) for w in sample)


def _pad_words(text: str, target_words: int, rng: np.random.Generator, formal: bool) -> str:
    formal_fillers = [
        "This pattern supports a stable interpretation",
        "the wording is internally consistent",
        "the main clue remains visible",
        "the answer follows from the described evidence",
    ]
    informal_fillers = [
        "I may be missing a detail",
        "that is just the sign I would use",
        "the clue is pretty direct",
        "it does not look very ambiguous",
    ]
    fillers = formal_fillers if formal else informal_fillers
    words = text.split()
    while len(words) < target_words:
        words.extend(str(rng.choice(fillers)).split())
    return " ".join(words[: max(target_words, 4)])


def _adversarial_disguise(text: str, rng: np.random.Generator) -> str:
    replacements = {
        "plausible": "reasonable",
        "coherent": "clear",
        "balanced": "even",
        "consistently": "often",
        "therefore": "so",
        "notably": "noticeably",
    }
    words = [replacements.get(w.strip(".,;:").lower(), w) for w in text.split()]
    if len(words) > 8:
        swap = int(rng.integers(2, len(words) - 2))
        words[swap], words[swap + 1] = words[swap + 1], words[swap]
    if rng.random() < 0.5:
        words.append("teh")
    return " ".join(words)


def _style_imitation(text: str, rng: np.random.Generator) -> str:
    lowered = text.replace("Overall,", "I think").replace("Based on the provided information,", "My guess is")
    lowered = lowered.replace("The explanation remains", "The reason feels")
    if rng.random() < 0.5:
        lowered += " I would still check the original item."
    return lowered


def _adaptive_rewrite(text: str, rng: np.random.Generator) -> str:
    remove = {"overall,", "therefore", "moreover", "notably", "balanced,", "coherent,"}
    words = [w for w in text.split() if w.lower().strip(".,;:") not in remove]
    if rng.random() < 0.4:
        words = words[::2] + words[1::2]
    return " ".join(words)

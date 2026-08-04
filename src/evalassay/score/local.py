"""Score options by exact log-likelihood under a local causal language model.

Options are scored, not generated. The model is asked, for each option in turn,
how likely that continuation is given the prompt, and the highest wins. This
matters for an audit in a way it does not for a leaderboard:

- **It is deterministic.** One forward pass per option, no sampling, no decoding
  parameters, no temperature. Two runs of the same audit on the same machine
  produce byte-identical numbers, which is what lets the run manifest promise
  reproducibility rather than merely hope for it.
- **It cannot fail to parse.** Generation-based scoring has to extract a choice
  from free text, and every extraction heuristic silently mislabels some
  answers. Those mislabellings are not spread evenly across conditions, so they
  land in the decomposition as artifacts that belong to the parser.

Scores are length-normalised by default. Summed log-likelihood grows more
negative with every token, so an unnormalised score systematically prefers short
options - and this tool exists partly to measure length artifacts, so building
one into the scorer would be self-defeating.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from evalassay.types import Item

FloatArray = NDArray[np.float64]

PROMPT_TEMPLATE: Final = "Question: {question}\nAnswer:"
"""Prompt for an item with a question."""

BLIND_TEMPLATE: Final = "Answer:"
"""Prompt for an item whose question has been hidden.

Deliberately the same shape minus the question, so the blind condition differs
from the normal one in exactly one respect.
"""

LABELS: Final = "ABCDEFGH"
"""Option labels used by the labelled scoring style."""

CLOZE: Final = "cloze"
"""Score each option as a continuation, with the options never shown as a list."""

LABELLED: Final = "labelled"
"""Present the options as a labelled list and score the label tokens."""

DTYPES: Final = ("float32", "bfloat16", "float16")
"""Weight precisions this backend will load.

``float32`` is the default because it is what every CPU supports well and it is
the least surprising. ``bfloat16`` halves the memory a model occupies, which is
the difference between a 1.5 billion parameter model fitting alongside other
work and not fitting at all. Precision is recorded in the scorer identity, so
two runs at different precisions can never be silently compared.
"""

MEMORY_FLOOR_MB: Final = 1500
"""Refuse to load a model when less free memory than this remains.

Loading a model that does not fit does not fail cleanly - it drives the machine
into swapping and takes every other process down with it. This tool is often run
on a laptop that is doing other things, so it checks before it allocates rather
than discovering the problem by making the machine unusable.
"""

CORES_LEFT_FREE: Final = 4
"""Logical processors deliberately not used, so the machine stays responsive.

Scoring is embarrassingly parallel and torch will take every core it is given.
On a machine doing other work that is the difference between a slow audit and an
unusable desktop, and an audit that finishes an hour later is a far smaller cost
than one that makes everything else stop.
"""

STYLES: Final = (CLOZE, LABELLED)
"""The two ways a local model can be asked a multiple-choice question.

The distinction is not cosmetic and it changes what an audit can see.

Under ``cloze`` an option is scored as a continuation of a prompt that never
contains the other options. Nothing about an option's *position* reaches the
model, so option permutation is structurally incapable of moving the result and
the audit reports it as inert.

Under ``labelled`` the options appear as a numbered list and the model scores the
label tokens. Position is now part of what the model sees, and a positional
preference becomes measurable - which is the setting most published leaderboard
numbers are produced in.

Auditing the same model on the same items under both styles is therefore a way
to measure how much of a score depends on the presentation rather than on the
question.
"""


def free_memory_mb() -> int | None:
    """Free physical memory, or ``None`` when it cannot be determined.

    Returns:
        Megabytes of free physical memory. ``None`` rather than a guess where
        the platform is not recognised, so a caller can tell "plenty" apart from
        "unknown" and not refuse to run on the strength of a fabricated number.
    """
    if os.name == "nt":

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys // (1024 * 1024))

    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:  # pragma: no cover - platform dependent
        return None
    try:  # pragma: no cover - platform dependent
        return int(sysconf("SC_AVPHYS_PAGES") * sysconf("SC_PAGE_SIZE") // (1024 * 1024))
    except (ValueError, OSError):
        return None


def default_thread_count() -> int:
    """How many threads to give torch, leaving the machine usable.

    Returns:
        At least one, and at most the processor count less
        :data:`CORES_LEFT_FREE`.
    """
    total = os.cpu_count() or 1
    return max(1, total - CORES_LEFT_FREE)


class LocalScorer:
    """Exact log-likelihood scoring with a locally loaded causal model.

    Attributes:
        model_name: Hugging Face model identifier or local path.
        length_normalise: Whether to divide by the option's token count.
    """

    def __init__(
        self,
        model_name: str,
        *,
        style: str = CLOZE,
        dtype: str = "float32",
        threads: int | None = None,
        length_normalise: bool = True,
        max_length: int = 1024,
        device: str = "cpu",
    ) -> None:
        """Load a model and tokenizer.

        Args:
            model_name: Hugging Face identifier or local path.
            style: One of :data:`STYLES`. ``cloze`` scores each option as a
                continuation and never shows the model the option list;
                ``labelled`` presents the options as a list and scores the label
                tokens. Only the second can exhibit positional effects.
            dtype: Weight precision, one of :data:`DTYPES`. ``bfloat16`` halves
                the memory a model occupies at a small cost in precision, and is
                recorded in the scorer identity so it cannot be silently mixed.
            threads: Torch thread count. Defaults to leaving
                :data:`CORES_LEFT_FREE` processors for everything else on the
                machine.
            length_normalise: Divide summed log-likelihood by token count.
            max_length: Truncation length for the combined prompt and option.
            device: Torch device string.

        Raises:
            ImportError: If the optional local-scoring dependencies are absent.
                Raised with an actionable message rather than a bare import
                error, because this is the most likely thing to be missing.
            ValueError: If the style or dtype is unrecognised, or too little
                memory is free to load a model safely.
        """
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {style!r}")
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")

        available = free_memory_mb()
        if available is not None and available < MEMORY_FLOOR_MB:
            raise ValueError(
                f"only {available} MB of memory is free and loading a model needs more; "
                f"refusing rather than driving the machine into swapping. Free some "
                f"memory, or load a smaller model, or pass dtype='bfloat16' to halve "
                f"what the weights occupy."
            )
        try:
            import torch  # noqa: PLC0415 - optional dependency, imported on demand
            from transformers import (  # noqa: PLC0415
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "local scoring needs the optional dependencies; install them with "
                'pip install "evalassay[local]"'
            ) from exc

        torch.set_num_threads(threads if threads is not None else default_thread_count())

        self.model_name = model_name
        self.style = style
        self.dtype = dtype
        self.length_normalise = length_normalise
        self._max_length = max_length
        self._torch = torch
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
        model.eval()
        model.to(device)
        self._model: Any = model
        self._device = device

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def scorer_id(self) -> str:
        """Identifier recorded in the run manifest."""
        suffix = "norm" if self.length_normalise else "sum"
        return f"local:{self.model_name}:{self.style}:{self.dtype}:{suffix}"

    @property
    def deterministic(self) -> bool:
        """Log-likelihood scoring involves no sampling."""
        return True

    def _prompt_for(self, item: Item) -> str:
        """Build the prompt for an item.

        Args:
            item: The item or variant.

        Returns:
            The prompt text.

        Raises:
            ValueError: If a labelled prompt would need more labels than exist.
        """
        question = item.question.strip()

        if self.style == CLOZE:
            return BLIND_TEMPLATE if not question else PROMPT_TEMPLATE.format(question=question)

        if item.n_choices > len(LABELS):
            raise ValueError(f"cannot label {item.n_choices} options; {len(LABELS)} available")
        lines = [f"Question: {question}"] if question else []
        lines.extend(f"{LABELS[i]}. {choice}" for i, choice in enumerate(item.choices))
        lines.append("Answer:")
        return "\n".join(lines)

    def _continuations(self, item: Item) -> list[str]:
        """The texts whose likelihood is compared, one per option.

        Args:
            item: The item or variant.

        Returns:
            The option texts under the cloze style, or the labels under the
            labelled style.
        """
        if self.style == CLOZE:
            return [" " + choice.strip() for choice in item.choices]
        return [" " + LABELS[index] for index in range(item.n_choices)]

    def score(self, item: Item) -> FloatArray:
        """Score every option by its likelihood as a continuation.

        All of an item's options share a prompt, so they are scored in a single
        padded forward pass rather than one pass each. That is not a micro
        optimisation: an audit evaluates each item under every coalition of
        interventions, so the option loop is the innermost of three and the
        saving compounds.

        Args:
            item: The item or variant. Only ``question`` and ``choices`` are
                read; the key is never consulted.

        Returns:
            One score per option, higher meaning more likely.
        """
        torch = self._torch
        prompt = self._prompt_for(item)
        prompt_ids: list[int] = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]

        sequences: list[list[int]] = []
        lengths: list[int] = []
        for text in self._continuations(item):
            continuation = self._tokenizer(text, add_special_tokens=False)["input_ids"]
            if not continuation:
                # An option that tokenises to nothing carries no evidence. Give
                # it one padding token and a neutral score rather than dividing
                # by zero, so one degenerate option cannot fail a whole run.
                continuation = [self._tokenizer.pad_token_id]
            combined = (prompt_ids + continuation)[-self._max_length :]
            sequences.append(combined)
            lengths.append(min(len(continuation), len(combined) - 1))

        width = max(len(sequence) for sequence in sequences)
        pad_id = self._tokenizer.pad_token_id
        padded = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            padded[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[row, : len(sequence)] = 1

        padded = padded.to(self._device)
        mask = mask.to(self._device)

        with torch.no_grad():
            logits = self._model(padded, attention_mask=mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = padded[:, 1:]
            gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        scores = np.empty(item.n_choices, dtype=np.float64)
        for row, sequence in enumerate(sequences):
            count = max(1, lengths[row])
            end = len(sequence) - 1
            total = float(gathered[row, end - count : end].sum().item())
            scores[row] = total / count if self.length_normalise else total
        return scores

    def describe(self) -> dict[str, Any]:
        """Configuration worth recording alongside a result.

        Returns:
            A JSON-serialisable description of the backend.
        """
        return {
            "backend": "local",
            "model": self.model_name,
            "style": self.style,
            "dtype": self.dtype,
            "threads": self._torch.get_num_threads(),
            "length_normalise": self.length_normalise,
            "max_length": self._max_length,
            "device": self._device,
        }

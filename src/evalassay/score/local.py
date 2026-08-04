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
            length_normalise: Divide summed log-likelihood by token count.
            max_length: Truncation length for the combined prompt and option.
            device: Torch device string.

        Raises:
            ImportError: If the optional local-scoring dependencies are absent.
                Raised with an actionable message rather than a bare import
                error, because this is the most likely thing to be missing.
            ValueError: If the style is not one of :data:`STYLES`.
        """
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {style!r}")
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

        self.model_name = model_name
        self.style = style
        self.length_normalise = length_normalise
        self._max_length = max_length
        self._torch = torch
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(model_name)
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
        return f"local:{self.model_name}:{self.style}:{suffix}"

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
            "length_normalise": self.length_normalise,
            "max_length": self._max_length,
            "device": self._device,
        }

"""Score options by asking a hosted model to choose one.

A hosted chat API does not expose per-token likelihoods, so this backend cannot
do what :mod:`evalassay.score.local` does. It presents the options as a labelled
list and constrains the reply to a single label.

Two consequences are recorded rather than glossed over:

- **It is not deterministic.** Even at temperature zero, a hosted model is a
  moving target: the same identifier can be served by different weights over
  time. The backend reports ``deterministic = False`` and the run manifest
  carries that through to the report, so nobody reads a reproducibility promise
  that was never made.
- **It produces a choice, not a ranking.** The score vector is one-hot. Every
  statistic in the audit is built from per-item correctness, so this is
  sufficient - but it means the audit cannot see how close a call was.

A reply that names no valid option is recorded as an abstention and scored as
incorrect, and the count is reported. Retrying until the model produces
something parseable would quietly select for the items it finds easy.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from evalassay.types import Item

FloatArray = NDArray[np.float64]

LABELS: Final = "ABCDEFGH"
"""Option labels presented to the model."""

SYSTEM_PROMPT: Final = (
    "You are answering a multiple-choice question. "
    "Reply with exactly one letter naming the best option, and nothing else."
)

MAX_LABELS: Final = len(LABELS)


class ApiScorer:
    """Choice scoring through a hosted chat model.

    Attributes:
        model: The model identifier passed to the API.
        abstentions: How many replies named no valid option.
    """

    def __init__(self, model: str, *, max_tokens: int = 4, client: Any | None = None) -> None:
        """Create the backend.

        Args:
            model: Model identifier.
            max_tokens: Reply length cap. A single letter is wanted, so this is
                small on purpose.
            client: An injected client, used by tests. When omitted, a client is
                constructed from the environment.

        Raises:
            ImportError: If the optional API dependency is absent and no client
                was injected.
        """
        if client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "API scoring needs the optional dependency; install it with "
                    'pip install "evalassay[api]"'
                ) from exc
            client = anthropic.Anthropic()

        self.model = model
        self._client = client
        self._max_tokens = max_tokens
        self.abstentions = 0

    @property
    def scorer_id(self) -> str:
        """Identifier recorded in the run manifest."""
        return f"api:{self.model}"

    @property
    def deterministic(self) -> bool:
        """A hosted model is not reproducible across time."""
        return False

    def _render(self, item: Item) -> str:
        """Render an item as a labelled multiple-choice prompt.

        Args:
            item: The item or variant.

        Returns:
            The prompt text.

        Raises:
            ValueError: If the item has more options than there are labels.
        """
        if item.n_choices > MAX_LABELS:
            raise ValueError(
                f"cannot label {item.n_choices} options; {MAX_LABELS} labels available"
            )

        lines = []
        question = item.question.strip()
        if question:
            lines.append(question)
        lines.extend(f"{LABELS[index]}. {choice}" for index, choice in enumerate(item.choices))
        return "\n".join(lines)

    def _parse(self, reply: str, n_choices: int) -> int | None:
        """Extract an option index from a reply.

        Args:
            reply: The model's text.
            n_choices: How many options were offered.

        Returns:
            The chosen index, or ``None`` if the reply named no valid option.
        """
        for character in reply.strip().upper():
            position = LABELS.find(character)
            if 0 <= position < n_choices:
                return position
        return None

    def score(self, item: Item) -> FloatArray:
        """Ask the model to choose, and return a one-hot score vector.

        Args:
            item: The item or variant. Only ``question`` and ``choices`` are
                read; the key is never consulted.

        Returns:
            One score per option: one for the chosen option, zero elsewhere. An
            abstention returns an all-zero vector, which the tie-break resolves
            deterministically and which is scored as incorrect unless it happens
            to land on the key.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._render(item)}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))

        scores = np.zeros(item.n_choices, dtype=np.float64)
        chosen = self._parse(text, item.n_choices)
        if chosen is None:
            self.abstentions += 1
            return scores
        scores[chosen] = 1.0
        return scores

    def describe(self) -> dict[str, Any]:
        """Configuration worth recording alongside a result.

        Returns:
            A JSON-serialisable description of the backend.
        """
        return {
            "backend": "api",
            "model": self.model,
            "max_tokens": self._max_tokens,
            "abstentions": self.abstentions,
            "deterministic": False,
        }

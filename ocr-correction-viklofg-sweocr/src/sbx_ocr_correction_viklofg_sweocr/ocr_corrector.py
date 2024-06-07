"""OCR Corrector."""

import re
from typing import List, Optional, Union

from parallel_corpus import graph
from sparv import api as sparv_api  # type: ignore [import-untyped]
from transformers import (  # type: ignore [import-untyped]
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    T5ForConditionalGeneration,
    pipeline,
)


def bytes_length(s: str) -> int:
    """Compute the length in bytes of the given str."""
    return len(s.encode("utf-8"))


TOK_SEP = " "
logger = sparv_api.get_logger(__name__)
TOKENIZER_REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
TOKENIZER_NAME = "google/byt5-small"
MODEL_REVISION = "84b138048992271be7617ccb11056bbcb9b72262"
MODEL_NAME = "viklofg/swedish-ocr-correction"

PUNCTUATION = re.compile(r"[.,:;!?]")


class OcrCorrector:
    """OCR Corrector."""

    TEXT_LIMIT: int = 127

    def __init__(
        self,
        *,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None,
        model: Optional[T5ForConditionalGeneration] = None,
    ) -> None:
        """Create a OCR corrector."""
        self.tokenizer = tokenizer or self._default_tokenizer()
        self.model = model or self._default_model()
        self.pipeline = pipeline(
            "text2text-generation", model=self.model, tokenizer=self.tokenizer
        )

    @classmethod
    def default(cls) -> "OcrCorrector":
        """Create a OCR Corrector with default tokenizer and model."""
        return cls(model=cls._default_model(), tokenizer=cls._default_tokenizer())

    @classmethod
    def _default_model(cls) -> T5ForConditionalGeneration:
        """Create the default model."""
        return T5ForConditionalGeneration.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    @classmethod
    def _default_tokenizer(cls) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """Create the default tokenizer."""
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)

    def calculate_corrections(self, text: List[str]) -> List[Optional[str]]:
        """Calculate corrections for the given text.

        Args:
            text (List[str]): The text as a list of strings

        Returns:
            List[Optional[str]]: A list of annotations or None
        """
        logger.debug("Analyzing '%s'", text)

        parts: List[str] = []
        curr_part: List[str] = []
        curr_len = 0
        ocr_corrections: List[Optional[str]] = []
        for word in text:
            len_word = bytes_length(word)
            if (curr_len + len_word + 1) > self.TEXT_LIMIT:
                parts.append(TOK_SEP.join(curr_part))
                curr_part, curr_len = [word], len_word
            else:
                curr_part.append(word)
                curr_len = len_word if curr_len == 0 else curr_len + len_word + 1
        if len(curr_part) > 0:
            parts.append(TOK_SEP.join(curr_part))
        for part in parts:
            graph_initial = graph.init(part)
            suggested_text = self.pipeline(part)[0]["generated_text"]

            suggested_text = PUNCTUATION.sub(r" \0", suggested_text)
            graph_aligned = graph.set_target(graph_initial, suggested_text)
            ocr_corrections.extend(align_and_diff(graph_aligned))

        logger.debug("Finished analyzing. ocr_corrections=%s", ocr_corrections)
        return ocr_corrections


def align_and_diff(g: graph.Graph) -> List[Optional[str]]:
    """Align and diff changes in the graph.

    Args:
        g (graph.Graph): the graph to work with

    Raises:
        NotImplementedError: if there are several sources

    Returns:
        List[Optional[str]]: list of corrections or None
    """
    corrections = []
    edge_map = graph.edge_map(g)
    for s_token in g.source:
        edge = edge_map[s_token.id]

        source_ids = [id_ for id_ in edge.ids if id_.startswith("s")]
        target_ids = [id_ for id_ in edge.ids if id_.startswith("t")]
        if len(source_ids) == len(target_ids):
            source_text = "".join(
                lookup_text(g, s_id, graph.Side.source) for s_id in source_ids
            ).strip()
            target_text = "".join(
                lookup_text(g, s_id, graph.Side.target) for s_id in target_ids
            ).strip()
            corrections.append(target_text if source_text != target_text else None)

        elif len(source_ids) == 1:
            target_texts = " ".join(
                lookup_text(g, id_, graph.Side.target).strip() for id_ in target_ids
            )
            source_text = s_token.text.strip()
            corrections.append(target_texts if source_text != target_texts else None)
        elif len(target_ids) == 1:
            # TODO Handle this correct (https://github.com/spraakbanken/sparv-sbx-ocr-correction/issues/44)
            logger.warning(
                "Handle several sources, see https://github.com/spraakbanken/sparv-sbx-ocr-correction/issues/44, source_ids=%s target_ids=%s g.source=%s g.target=%s",  # noqa: E501
                source_ids,
                target_ids,
                g.source,
                g.target,
            )
            target_text = lookup_text(g, target_ids[0], graph.Side.target).strip()
            corrections.append(target_text)
        else:
            # TODO Handle this correct (https://github.com/spraakbanken/sparv-sbx-ocr-correction/issues/44)
            raise NotImplementedError(
                f"Handle several sources, {source_ids=} {target_ids=} {g.source=} {g.target=}"
            )

    return corrections


def lookup_text(g: graph.Graph, id_: str, side: graph.Side) -> str:
    """Lookup text in graph for given id and side.

    Args:
        g (graph.Graph): the graph to work with
        id_ (str): the id to  search for
        side (graph.Side): the side to look at

    Raises:
        KeyError: if the id is not found

    Returns:
        str: the text for the id
    """
    if side == graph.Side.source:
        for token in g.source:
            if token.id == id_:
                return token.text
    else:
        for token in g.target:
            if token.id == id_:
                return token.text
    raise KeyError(
        f"The id={id_} isn't found in the given graph on side={side}",
    )

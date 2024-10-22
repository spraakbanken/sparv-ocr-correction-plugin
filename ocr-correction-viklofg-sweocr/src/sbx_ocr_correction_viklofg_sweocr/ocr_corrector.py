"""OCR Corrector."""

import re
from typing import List, Optional, Tuple, Union

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

PUNCTUATION = re.compile(r"([.,:;!?])")


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

    def calculate_corrections(
        self, text: List[str]
    ) -> List[Tuple[Tuple[int, int], Optional[str]]]:
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
        ocr_corrections: List[Tuple[Tuple[int, int], Optional[str]]] = []
        text_chunks = []
        start_i = 0
        for word_i, word in enumerate(text):
            len_word = bytes_length(word)
            if (curr_len + len_word + 1) > self.TEXT_LIMIT:
                parts.append(self._build_part(curr_part))
                text_chunks.append(text[start_i:word_i])
                curr_part, curr_len = [word], len_word
                start_i = word_i
            else:
                curr_part.append(word)
                curr_len = len_word if curr_len == 0 else curr_len + len_word + 1
        if len(curr_part) > 0:
            parts.append(self._build_part(curr_part))
            text_chunks.append(text[start_i:])
        logger.debug("split text in %d chunks", len(parts))
        ctx = AlignDiffContext()
        for part_i, part in enumerate(parts):
            logger.debug("processing chunk %d: '%s'", part_i, part)
            # graph_initial = graph.init(part)
            suggested_text = self.pipeline(part)[0]["generated_text"]
            print(f"{suggested_text=}")
            # suggested_text = PUNCTUATION.sub(r" \0", suggested_text)
            # graph_aligned = graph.init_from_source_and_target(part, suggested_text)
            # graph_aligned = graph.set_target(graph_initial, suggested_text)
            # ocr_corrections.extend(align_and_diff(graph_aligned))
            suggested_text_tokenized = re.sub(r"([.,:;!?])", r" \1", suggested_text)
            # text_vs_suggested = graph.init(TOK_SEP.join(text_chunks[part_i]))
            # split = ""
            # for sugg in suggested_text.split(" "):
            #     for sugg_part in PUNCTUATION.split(sugg):
            #         suggested_text_tokenized += f"{split}{sugg_part}"
            #         split = " "
            # print(
            #     f"text_vs_suggested.source='{''.join(graph.get_side_texts(text_vs_suggested,
            #  graph.Side.source))}'"
            # )
            # print(f"{suggested_text_tokenized=}")
            text_vs_suggested_aligned = graph.init_with_source_and_target(
                TOK_SEP.join(text_chunks[part_i]), suggested_text_tokenized
            )
            print(
                "text_vs_suggested.source='{}'".format(
                    "".join(graph.get_side_texts(text_vs_suggested_aligned, graph.Side.source))
                )
            )
            print(f"{suggested_text_tokenized=}")
            print(f"{text_vs_suggested_aligned=}")
            logger.debug("text_vs_suggested_aligned=%s", text_vs_suggested_aligned)
            ocr_corrections.extend(align_and_diff(text_vs_suggested_aligned, ctx=ctx))

        logger.debug("Finished analyzing. ocr_corrections=%s", ocr_corrections)
        return ocr_corrections

    @classmethod
    def _build_part(cls, parts: List[str]) -> str:
        if not parts:
            return ""
        out = parts[0]
        for part in parts[1:]:
            if part.isalnum():
                out = f"{out} {part}"
            else:
                out += part
        return out


class AlignDiffContext:
    """Context for align_and_diff."""

    def __init__(self) -> None:
        """Create the Context."""
        self._curr_token: int = 0
        self._skip_if_positive = 0

    def make_span(self, length: int = 1) -> Tuple[int, int]:
        """Create a span of given length.

        Args:
            length (int, optional): the length of the span. Defaults to 1.

        Returns:
            Tuple[int, int]: the created span of given length
        """
        start_pos = self._curr_token
        self._curr_token += length
        self._skip_if_positive = length - 1
        end_pos = self._curr_token
        return (start_pos, end_pos)

    def skip_token(self) -> bool:
        """Return true if this token should be skipped.

        Returns:
            bool: True if this token should be skipped
        """
        if self._skip_if_positive > 0:
            self._skip_if_positive -= 1
            return True
        return False


def align_and_diff(
    g: graph.Graph, *, ctx: AlignDiffContext
) -> List[Tuple[Tuple[int, int], Optional[str]]]:
    """Align and diff changes in the graph.

    Args:
        g (graph.Graph): the graph to work with
        ctx (AlignDiffContext): the context for using this function

    Raises:
        NotImplementedError: if there are several sources

    Returns:
        List[Optional[str]]: list of corrections or None
    """
    logger.info("align and diff the graph chunk")
    corrections = []
    edge_map = graph.edge_map(g)
    for s_i, s_token in enumerate(g.source):
        print(f"{s_i=} {s_token=}")
        logger.debug("s_i=%s s_token=%s", s_i, s_token)
        edge = edge_map[s_token.id]

        source_ids = [id_ for id_ in edge.ids if id_.startswith("s")]
        target_ids = [id_ for id_ in edge.ids if id_.startswith("t")]
        print(f"{source_ids=} {target_ids=}")
        if ctx.skip_token():
            print(f"Skipping {s_i=} {s_token=}")
            continue

        logger.debug("source_ids=%s target_ids=%s", source_ids, target_ids)
        source_text = "".join(
            lookup_text(g, s_id, graph.Side.source) for s_id in source_ids
        ).strip()
        target_text = "".join(
            lookup_text(g, s_id, graph.Side.target) for s_id in target_ids
        ).strip()
        text_diff_opt = target_text if source_text != target_text else None
        print(f"{source_text=} {target_text=}")
        logger.debug("source_text=%s target_text=%s", source_text, target_text)
        if len(source_ids) == 1:
            corrections.append((ctx.make_span(), text_diff_opt))
        elif len(target_ids) == 1:
            corrections.append((ctx.make_span(len(source_ids)), text_diff_opt))
        # if len(source_ids) == len(target_ids):
        #     source_text = "".join(
        #         lookup_text(g, s_id, graph.Side.source) for s_id in source_ids
        #     ).strip()
        #     target_text = "".join(
        #         lookup_text(g, s_id, graph.Side.target) for s_id in target_ids
        #     ).strip()
        #     corrections.append(target_text if source_text != target_text else None)

        # elif len(source_ids) == 1:
        #     target_texts = " ".join(
        #         lookup_text(g, id_, graph.Side.target).strip() for id_ in target_ids
        #     )
        #     source_text = s_token.text.strip()
        #     corrections.append(target_texts if source_text != target_texts else None)
        # elif len(target_ids) == 1:
        #     # TODO Handle this correct (https://github.com/spraakbanken/sparv-sbx-ocr-correction/issues/44)
        #     logger.warning(
        #         "Handle several sources, see https://github.com/spraakbanken/sparv-sbx-ocr-correction/issues/44, source_ids=%s target_ids=%s g.source=%s g.target=%s",  # noqa: E501
        #         source_ids,
        #         target_ids,
        #         g.source,
        #         g.target,
        #     )
        #     target_text = lookup_text(g, target_ids[0], graph.Side.target).strip()
        #     corrections.append(target_text)
        else:
            # TODO Handle this correct (https://github.com/spraakbanken/sparv-sbx-ocr-correction/issues/44)
            raise NotImplementedError(
                f"Handle several sources, {source_ids=} {target_ids=} {g.source=} {g.target=}"
            )
    print(f"returning {corrections=}")
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

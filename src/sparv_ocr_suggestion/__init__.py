from sparv.api import (  # type: ignore [import-untyped]
    annotator,
    Output,
    get_logger,
    Annotation,
    Config,
    SparvErrorMessage,
)

from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer

__description__ = "Calculating word neighbours by mask a word in a BERT model."


__config__ = [
    Config(
        "sparv_ocr_suggestion.model",
        description="Huggingface pretrained model name",
        default="viklofg/swedish-ocr-correction",
    ),
    Config(
        "sparv_ocr_suggestion.tokenizer",
        description="HuggingFace pretrained tokenizer name",
        default="google/byt5-small",
    ),

]

__version__ = "0.2.1"

logger = get_logger(__name__)

TOK_SEP = " "




ocr = 'Den i HandelstidniDgens g&rdagsnnmmer omtalade hvalfisken, sorn fångats i Frölnndaviken'

@annotator(
    "Word neighbour tagging with a masked Bert model",
)
def annotate_ocr_suggestion(
    out_ocr_suggestion: Output = Output(
        "<token>:sparv_ocr_suggestion.ocr-suggestion",
        cls="ocr_suggestion",
        description="Transformer neighbours from masked BERT (format: '|<word>:<score>|...|)",
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    model_name: str = Config("sparv_ocr_suggestion.model"),
    tokenizer_name: str = Config("sparv_ocr_suggestion.tokenizer"),
) -> None:
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    ocr_suggestor = OCRSuggestor(model=model, tokenizer=tokenizer)

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    out_ocr_suggestion_annotation = word.create_empty_attribute()
    logger.warn("%d",len(out_ocr_suggestion))

    logger.progress(total=len(sentences))  # type: ignore
    for sent in sentences:
        logger.progress()  # type: ignore
        sent_to_tag = TOK_SEP.join(
            token_word[token_index]
            for token_index in sent
        )

        ocr_suggestions = ocr_suggestor.calculate_suggestions(
            sent_to_tag
        )
        annotations = ocr_suggestions.replace(",", " ,")
        annotations = annotations.split(TOK_SEP)
        logger.warn("annotations (%d) '%s'", len(annotations), annotations)
        out_ocr_suggestion_annotation[:] = annotations

    logger.info("writing annotations")
    out_ocr_suggestion.write(out_ocr_suggestion_annotation)


class OCRSuggestor:
    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pipeline = pipeline('text2text-generation',model=model, tokenizer=tokenizer)

    def calculate_suggestions(self, text: str) -> str:
        logger.warn("Analyzing '%s'", text)
        suggested_text = self.pipeline(text)
        logger.warn("Output: '%s'", suggested_text)
        return suggested_text[0]["generated_text"]

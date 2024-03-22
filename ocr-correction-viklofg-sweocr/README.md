# sparv-sbx-ocr-correction

[![PyPI version](https://badge.fury.io/py/sparv-sbx-ocr-correction.svg)](https://pypi.org/project/sparv-sbx-ocr-correction)

[![Maturity badge - level 2](https://img.shields.io/badge/Maturity-Level%202%20--%20First%20Release-yellowgreen.svg)](https://github.com/spraakbanken/getting-started/blob/main/scorecard.md)
[![Stage](https://img.shields.io/pypi/status/sparv-sbx-ocr-correction)](https://pypi.org/project//sparv-sbx-ocr-correction)

Sparv plugin to annotate corrections to OCR:ed documents.

## Install

In a virtual environment:

```bash
pip install sparv-sbx-ocr-correction
```

or if you have `sparv` installed with `pipx`:

```bash
pipx inject sparv-pipeline sparv-sbx-ocr-correction
```

## Metadata

### Model

Type | HuggingFace Model | Revision
--- | --- | ---
Model | [`viklofg/swedish-ocr-correction`](https://huggingface.co/viklofg/swedish-ocr-correction) | 84b138048992271be7617ccb11056bbcb9b72262
Tokenizer | [`google/byt5-small`](https://huggingface.co/google/byt5-small) | 68377bdc18a2ffec8a0533fef03b1c513a4dd49d

## Changelog

This project keeps a [changelog](./CHANGELOG.md).

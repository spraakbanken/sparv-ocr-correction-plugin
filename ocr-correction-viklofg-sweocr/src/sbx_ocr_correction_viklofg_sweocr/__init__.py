"""Sparv plugin to annotate tokens with correction of OCR errors."""

from sbx_ocr_correction_viklofg_sweocr.annotations import annotate_ocr_correction

__all__ = ["annotate_ocr_correction"]

__description__ = "Annotate tokens with corrections of OCR-errors."

__version__ = "0.4.2"

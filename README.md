# Receipt OCR & Structured Expense Extraction

Production-style OCR pipeline developed for the Carbon Crunch ML Ops
internship shortlisting assignment.

## Status

Project initialization and dataset profiling completed.

Current environment:

- Windows 11
- Python 3.12
- CPU-only
- Intel Core i5-1155G7
- 8 GB RAM
- 371 receipt images

## Objective

Build a robust system that:

1. Preprocesses receipt images.
2. Performs OCR.
3. Extracts structured receipt information.
4. Calculates field-level confidence.
5. Detects ambiguous or conflicting results.
6. Generates financial summaries.
7. Evaluates the OCR and extraction pipeline.

## Dataset

The dataset contains 371 readable receipt images.

The dataset includes diverse receipt layouts, image qualities,
fonts, and document formats.

The full dataset is intentionally not committed to Git.

## Planned Pipeline

Receipt Image
→ Image Validation
→ Quality Assessment
→ Preprocessing
→ OCR
→ Text/Layout Processing
→ Field Extraction
→ Validation
→ Confidence Scoring
→ Conflict Detection
→ Structured JSON
→ Financial Summary
→ Evaluation

## Project Structure

```text
configs/
data/
docs/
notebooks/
outputs/
scripts/
src/
tests/
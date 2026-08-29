# Carbon Crunch — Receipt OCR & Structured Expense Extraction

A production-oriented OCR and structured expense extraction pipeline developed for the **Carbon Crunch ML Ops Internship Assignment**.

The system processes receipt images using **PaddleOCR**, extracts structured receipt and financial information using a rule-based parser, calculates OCR and extraction confidence, detects potential ambiguities, and generates machine-readable JSON and consolidated financial summaries.

---

## Project Status

**Status: Completed**

| Metric                        |                Result |
| ----------------------------- | --------------------: |
| Receipt images processed      |               **371** |
| Successfully processed        |               **371** |
| Failed                        |                 **0** |
| Processing success rate       |              **100%** |
| Receipts with extracted total |               **348** |
| Total expense extracted       |         **28,154.92** |
| Average receipt total         |             **80.90** |
| OCR engine                    |     **PaddleOCR 3.x** |
| Inference backend             |      **ONNX Runtime** |
| Execution environment         |        **Kaggle CPU** |
| Per-receipt JSON outputs      |               **371** |
| Summary formats               | **CSV + JSON + XLSX** |

> **Important:** A successful processing status means that the pipeline completed without a processing-level exception. It does not mean that every individual receipt field was extracted perfectly. Low-confidence or incomplete receipts may require manual review.

---

# 1. Objective

The objective of this project is to build a robust receipt-processing pipeline capable of converting receipt images into structured and machine-readable expense information.

The pipeline is designed to:

1. Discover and validate receipt images.
2. Perform OCR on receipt images.
3. Extract recognized text and OCR confidence scores.
4. Identify important receipt fields.
5. Extract purchased items where possible.
6. Extract subtotal, discount, tax, and total.
7. Identify payment methods.
8. Calculate OCR confidence.
9. Calculate extraction confidence.
10. Generate warnings for uncertain or missing information.
11. Generate one structured JSON file per receipt.
12. Generate consolidated expense summaries.
13. Generate processing statistics for reproducibility.

---

# 2. End-to-End Architecture

```text
                    Receipt Images
                         │
                         ▼
              Image Discovery & Validation
                         │
                         ▼
                    PaddleOCR
              ┌──────────┴──────────┐
              │                     │
        Text Detection        Text Recognition
              │                     │
              └──────────┬──────────┘
                         ▼
                  OCR Result Data
                         │
                         ▼
                OCR Normalization
                         │
                         ▼
              Rule-Based Receipt Parser
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Receipt Fields     Line Items     Financial Fields
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                Confidence Scoring
                         │
                         ▼
                 Validation & Warnings
                         │
                         ▼
                Structured JSON Output
                         │
                         ▼
              Consolidated Expense Data
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
            CSV         JSON        XLSX
```

---

# 3. Dataset

The final pipeline was executed on a dataset containing:

* **371 receipt images**
* Different receipt layouts
* Different text sizes and fonts
* Different image qualities
* Different receipt structures
* Different financial field arrangements

The original dataset is **not committed to this GitHub repository**.

This keeps the repository lightweight and avoids unnecessary redistribution of the source dataset.

The receipt dataset is supplied separately during execution, including through the Kaggle environment used for the final processing run.

---

# 4. Technology Stack

## Programming Language

* Python 3.12

## OCR

* PaddleOCR 3.x
* PaddleX OCR pipeline
* ONNX Runtime

## Data Processing

* Pandas
* Python Standard Library
* Regular Expressions
* JSON
* CSV

## Output Generation

* JSON
* CSV
* Excel/XLSX
* OpenPyXL

## Image Processing

* Pillow

## Utilities

* tqdm

## Execution Environment

The final 371-receipt processing run was performed in a:

```text
Kaggle CPU environment
```

The repository itself is designed to be reproducible outside Kaggle when the required dependencies and receipt dataset are available.

---

# 5. OCR Approach

The project uses **PaddleOCR** for receipt text detection and recognition.

The final execution uses CPU inference with unnecessary document-processing components disabled.

The OCR configuration used by the pipeline includes settings equivalent to:

```python
COMMON_KWARGS = dict(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=960,
    text_det_limit_type="max",
    text_rec_score_thresh=0.30,
    device="cpu",
)
```

The pipeline attempts to initialize PaddleOCR using the ONNX Runtime backend:

```python
PaddleOCR(
    engine="onnxruntime",
    **COMMON_KWARGS
)
```

Using ONNX Runtime avoids dependency on the Paddle Inference execution path that caused compatibility issues during development.

---

# 6. OCR Result Processing

PaddleOCR 3.x returns structured OCR results.

The pipeline extracts information such as:

```text
OCRResult
    └── json
         └── res
              ├── rec_texts
              ├── rec_scores
              ├── rec_polys
              └── rec_boxes
```

For each receipt, the pipeline preserves:

* Recognized text
* OCR confidence scores
* Text bounding boxes
* Polygon information where available
* Raw OCR text

Example OCR text may look like:

```text
WAL*MART
SUPERCENTER
BANANAS
0.41 1b
FRAP
5.48 N
DISCOUNT GIVEN
0.57
SUBTOTAL
5.11
TOTAL
5.11
CASH TEND
11.00
CHANGE DUE
5.89
```

The normalized OCR output is then passed to the receipt parser.

---

# 7. Structured Receipt Extraction

A rule-based parser is used to transform OCR text into structured receipt information.

The parser attempts to extract:

```text
merchant
receipt_date
receipt_number
items
subtotal
discount
tax
total
payment_method
ocr_confidence
extraction_confidence
raw_text
warnings
```

The approach is intentionally explainable and deterministic.

---

# 8. Merchant Extraction

Merchant identification is based on positional and text-based heuristics.

The parser primarily examines the upper portion of the receipt while filtering common non-merchant information such as:

```text
phone
address
manager
hours
customer
website
thank you
```

This reduces the possibility of incorrectly identifying a telephone number, address, or promotional message as the merchant name.

---

# 9. Date Extraction

The parser supports common date formats, including patterns such as:

```text
08/20/10
2025-08-20
20-08-2025
Aug 20 2025
20 Aug 2025
```

Date extraction also considers contextual labels such as:

```text
date
invoice
receipt
transaction
```

---

# 10. Receipt / Transaction Number Extraction

The parser searches for identifiers associated with:

```text
receipt
transaction
txn
trans
order
invoice
```

Examples include:

```text
Receipt No
Transaction #
TXN
Order No
Invoice No
```

---

# 11. Item Extraction

The parser attempts to identify purchasable line items from OCR text.

A typical item structure is:

```json
{
  "name": "BANANAS",
  "quantity": 1.0,
  "unit_price": 0.41,
  "total_price": 0.41
}
```

The parser attempts to handle patterns involving:

* Quantity
* Weight
* Unit price
* Extended price
* Multiplication expressions

For example:

```text
2 x 5.99
```

can be interpreted as:

```text
quantity = 2
unit_price = 5.99
```

Because receipt layouts vary considerably, item extraction is treated as a confidence-based process rather than assuming that every OCR line represents a purchasable item.

---

# 12. Financial Field Extraction

Financial fields are extracted using field-specific patterns and contextual heuristics.

## Subtotal

The parser recognizes labels such as:

```text
SUBTOTAL
SUB TOTAL
MERCHANDISE SUBTOTAL
ITEM SUBTOTAL
```

## Discount

The parser recognizes labels such as:

```text
DISCOUNT
DISCOUNTS
SAVINGS
COUPON
PROMOTION
PROMO
```

## Tax

The parser recognizes labels such as:

```text
TAX
SALES TAX
GST
CGST
SGST
IGST
VAT
```

## Total

The parser recognizes labels such as:

```text
TOTAL
GRAND TOTAL
TOTAL DUE
AMOUNT DUE
BALANCE DUE
ORDER TOTAL
INVOICE TOTAL
PAYABLE
AMOUNT PAYABLE
```

Special handling is used for:

```text
CASH TEND
CHANGE DUE
```

so that payment tender and change values are not incorrectly interpreted as the final receipt total.

---

# 13. Payment Method Detection

The parser checks for common payment methods, including:

```text
GOOGLE PAY
PHONEPE
PAYTM
BHIM UPI
UPI
VISA
MASTERCARD
AMERICAN EXPRESS
CREDIT CARD
DEBIT CARD
CASH
CARD
```

The detected value is stored in the structured receipt JSON when sufficiently identifiable.

---

# 14. Confidence Scoring

The pipeline maintains two primary confidence measurements.

## OCR Confidence

OCR confidence is derived from the recognition confidence scores returned by PaddleOCR.

Conceptually:

```text
OCR Confidence
=
Average recognition confidence of detected text
```

The result is normalized before being stored in the structured output.

## Extraction Confidence

Extraction confidence represents the confidence in the structured receipt interpretation.

It considers signals such as:

* OCR confidence
* Merchant detection
* Date detection
* Receipt number detection
* Subtotal detection
* Tax detection
* Total detection
* Payment method detection
* Item extraction
* Processing warnings

This allows downstream systems to distinguish between receipts that were processed successfully and receipts that may need manual review.

---

# 15. Warning System

The pipeline generates warnings when important information cannot be confidently extracted.

Examples include:

```text
Merchant could not be confidently identified.

Receipt date could not be identified.

Receipt total could not be confidently identified.

No purchasable items could be confidently extracted.

Overall OCR confidence is low; manual review recommended.

OCR confidence is moderate; extracted fields should be reviewed.
```

Warnings are preserved in the per-receipt JSON output.

---

# 16. Output Structure

The generated submission package follows this structure:

```text
carbon_crunch_submission/

├── processing_report.json
│
├── expense_summary.csv
├── expense_summary.json
├── expense_summary.xlsx
│
├── json/
│   ├── 0.json
│   ├── 1.json
│   ├── 2.json
│   ├── ...
│   └── X51005806719.json
│
└── ocr/
    ├── 0_ocr.json
    ├── 1_ocr.json
    ├── 2_ocr.json
    ├── ...
    └── X51005806719_ocr.json
```

The actual receipt filenames are preserved where applicable rather than being renamed to generic sequential receipt names.

---

# 17. Per-Receipt JSON

Each receipt receives an individual structured JSON file.

Example:

```json
{
  "merchant": "WAL*MART",
  "receipt_date": "08/20/10",
  "receipt_number": "03178",
  "items": [],
  "subtotal": 5.11,
  "discount": 0.57,
  "tax": null,
  "total": 5.11,
  "payment_method": "CASH",
  "ocr_confidence": 0.97,
  "extraction_confidence": 0.802,
  "raw_text": "...",
  "warnings": [
    "No purchasable items could be confidently extracted."
  ]
}
```

> The exact fields and values vary according to the information available in each receipt.

---

# 18. Raw OCR Outputs

Raw OCR results are stored separately from the structured parser outputs.

This provides an audit/debugging layer that allows extraction errors to be investigated without rerunning OCR.

A raw OCR output may contain:

```json
{
  "receipt_id": "0",
  "source_image": "0.jpg",
  "ocr_confidence": 0.97,
  "lines": [
    {
      "text": "WAL*MART",
      "confidence": 0.9763
    },
    {
      "text": "TOTAL",
      "confidence": 0.9999
    }
  ],
  "raw_text": "WAL*MART\nTOTAL\n..."
}
```

Where available, OCR bounding-box information is also preserved.

---

# 19. Expense Summary

The pipeline generates three consolidated financial summary formats:

```text
expense_summary.csv
expense_summary.json
expense_summary.xlsx
```

The receipt-level summary can contain fields such as:

* Receipt ID
* Source image
* Merchant
* Receipt date
* Receipt number
* Subtotal
* Discount
* Tax
* Total
* Payment method
* Item count
* OCR confidence
* Extraction confidence
* Warning count

---

# 20. Final Financial Summary

The final Kaggle execution produced:

```text
Receipts processed       : 371
Receipts with total      : 348
Total expense            : 28154.92
Average receipt total    : 80.90
```

The remaining receipts were successfully processed at the pipeline level but did not have a confidently extracted final total.

This distinction is important:

```text
Processing success ≠ Complete field extraction
```

A receipt can be successfully processed while individual fields remain unavailable or uncertain.

---

# 21. Processing Report

The pipeline generates:

```text
processing_report.json
```

to record processing-level statistics.

The final run reported:

```json
{
  "total_images_found": 371,
  "successful": 371,
  "failed": 0,
  "success_rate_percent": 100.0,
  "json_files_created": 371
}
```

The actual report file included with the final deliverables should be treated as the authoritative record of the execution.

---

# 22. Project Structure

The GitHub repository is organized as follows:

```text
carbon-crunch-receipt-ocr/
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── interim/
│   │   └── .gitkeep
│   ├── parser_results.csv
│   ├── processed/
│   │   └── .gitkeep
│   ├── raw/
│   │   └── .gitkeep
│   └── sample/
│       └── .gitkeep
│
├── outputs/
│   ├── expense_summary.csv
│   ├── expense_summary.json
│   ├── expense_summary.xlsx
│   ├── json/
│   │   └── Per-receipt structured JSON files
│   ├── logs/
│   │   └── .gitkeep
│   ├── ocr/
│   │   └── Raw OCR JSON files
│   ├── processing_report.json
│   ├── reports/
│   │   └── .gitkeep
│   └── visualizations/
│       └── .gitkeep
│
├── pyproject.toml
├── requirements.txt
│
├── scripts/
│   ├── analyze_dataset.py
│   ├── evaluate_parser.py
│   ├── run_pipeline.py
│   ├── test_all_receipts.py
│   ├── test_confidence.py
│   ├── test_ocr.py
│   ├── test_parser.py
│   ├── test_summary.py
│   ├── test_tesseract.py
│   └── test_validation.py
│
└── src/
    └── receipt_ocr/
        ├── __init__.py
        │
        ├── confidence/
        │   ├── __init__.py
        │   └── confidence_scorer.py
        │
        ├── ocr/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── models.py
        │   ├── paddle_engine.py
        │   └── tesseract_engine.py
        │
        ├── parser/
        │   ├── __init__.py
        │   └── receipt_parser.py
        │
        ├── schemas.py
        │
        ├── summary/
        │   ├── __init__.py
        │   └── financial_summary.py
        │
        └── validation/
            ├── __init__.py
            ├── conflict_detector.py
            └── receipt_validator.py
```

---

# 23. Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd carbon-crunch-receipt-ocr
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 24. Requirements

The main dependencies include:

```text
paddlepaddle
paddleocr
onnxruntime
pandas
openpyxl
Pillow
tqdm
```

The exact dependency versions used for the project are specified in:

```text
requirements.txt
```

---

# 25. Running the Pipeline

The main pipeline can be executed using:

```bash
python scripts/run_pipeline.py
```

The input dataset should be placed in the directory configured by the project configuration.

For Kaggle execution, the input path should point to the mounted Kaggle dataset.

The final execution used the Kaggle environment and generated a submission package containing:

```text
README.md
requirements.txt
processing_report.json
expense_summary.csv
expense_summary.json
expense_summary.xlsx
json_outputs/
ocr_raw_outputs/
```

---

# 26. Testing and Validation Scripts

The repository contains several scripts for testing individual pipeline components:

```text
scripts/
├── test_all_receipts.py
├── test_confidence.py
├── test_ocr.py
├── test_parser.py
├── test_summary.py
├── test_tesseract.py
└── test_validation.py
```

There is also a parser evaluation script:

```text
scripts/evaluate_parser.py
```

These scripts support component-level debugging and validation.

---

# 27. Challenges Faced

## 27.1 OCR Quality Variation

Receipt images vary significantly in:

* Resolution
* Blur
* Lighting
* Font size
* Contrast
* Text orientation
* Layout

Therefore, OCR output cannot always be assumed to be perfect.

### Mitigation

The pipeline preserves OCR confidence scores and generates warnings for uncertain extraction results.

---

## 27.2 Different Receipt Layouts

Receipts do not follow one standardized structure.

For example:

```text
ITEM                 10.00
```

while another receipt may use:

```text
ITEM
2 x 5.00
```

### Mitigation

The parser uses multiple regular-expression patterns and contextual heuristics instead of relying on one fixed layout.

---

## 27.3 Financial Field Ambiguity

Receipts can contain multiple monetary values.

For example:

```text
SUBTOTAL       5.11
TOTAL          5.11
CASH TEND     11.00
CHANGE DUE     5.89
```

Simply selecting the largest or last monetary value could result in an incorrect total.

### Mitigation

The parser uses field-specific patterns, contextual matching, and positional heuristics.

Special handling is applied to:

```text
CASH TEND
CHANGE DUE
```

---

## 27.4 PaddleOCR Compatibility

During development, compatibility issues were encountered around the Paddle Inference execution path.

One observed issue involved:

```text
AnalysisConfig.set_optimization_level
```

### Mitigation

The final execution uses the ONNX Runtime backend where supported.

This reduced dependency on the problematic Paddle Inference execution path.

---

# 28. Improvements and Future Work

The current implementation provides a lightweight and explainable baseline.

Potential improvements include:

## 28.1 Image Preprocessing

Future versions could add:

* Deskewing
* Denoising
* Contrast enhancement
* Adaptive thresholding
* Perspective correction
* Automatic cropping
* Resolution enhancement

---

## 28.2 Layout-Aware Extraction

Bounding-box coordinates could be used more extensively to reconstruct the receipt layout.

This could improve:

* Item extraction
* Quantity-price association
* Column detection
* Tax extraction
* Total identification

---

## 28.3 Machine Learning Based Document Understanding

A document-understanding model could be introduced for more robust extraction of:

```text
merchant
date
receipt number
items
subtotal
discount
tax
total
payment method
```

Potential approaches include transformer-based document models and receipt-specific layout models.

---

## 28.4 Accounting Validation

A financial consistency check could validate:

```text
subtotal - discount + tax ≈ total
```

when the required fields are available.

Receipts failing this consistency check could be flagged for manual review.

---

## 28.5 Currency and Locale Support

Future versions could support multiple currencies and locale-specific formats such as:

```text
INR
USD
EUR
GBP
```

along with:

* Decimal separators
* Thousands separators
* Date formats
* Tax terminology

---

## 28.6 Human-in-the-Loop Review

Receipts with:

```text
low OCR confidence
low extraction confidence
missing total
conflicting financial fields
```

could automatically be placed into a manual verification workflow.

---

# 29. Design Decisions

## Why PaddleOCR?

PaddleOCR provides:

* Text detection
* Text recognition
* Recognition confidence
* Bounding-box information
* Document OCR capabilities

It is therefore well suited for receipt OCR without requiring a custom OCR model to be trained from scratch.

---

## Why Rule-Based Extraction?

Receipt layouts vary significantly, but many target fields have recognizable semantic labels such as:

```text
TOTAL
SUBTOTAL
TAX
DISCOUNT
DATE
```

Regular expressions and contextual heuristics provide an explainable baseline that is:

* Easy to debug
* Deterministic
* Lightweight
* Easy to extend

---

## Why JSON?

JSON provides a machine-readable structure suitable for:

* APIs
* Databases
* Analytics
* Data pipelines
* Downstream ML systems

---

# 30. Reproducibility

The repository contains the source code, configuration, dependency information, and execution scripts required to reproduce the pipeline.

The dataset itself is intentionally excluded from GitHub.

To reproduce the processing:

1. Clone the repository.
2. Create a Python environment.
3. Install the required dependencies.
4. Provide the receipt dataset.
5. Configure the input path.
6. Run the pipeline.
7. Inspect the generated JSON files.
8. Inspect the consolidated expense summaries.
9. Review warnings and confidence scores for uncertain receipts.

The final benchmark reported in this README corresponds to the **371-receipt Kaggle execution**.

---

# 31. Final Results

The final Kaggle execution processed the complete dataset successfully at the pipeline level:

```text
======================================================================
PROCESSING COMPLETE
======================================================================

Total images       : 371
Successful         : 371
Failed             : 0
Success rate       : 100.0%
JSON files         : 371
Processing time    : 1703.62 seconds
```

Financial extraction:

```text
Receipts with total : 348
Total expense       : 28154.92
Average receipt     : 80.90
```

The generated submission package contained:

```text
README.md
expense_summary.csv
expense_summary.json
expense_summary.xlsx
json_outputs/
ocr_raw_outputs/
processing_report.json
requirements.txt
```

---

# 32. Important Interpretation of Results

The reported **100% processing success rate** should be interpreted as:

> All 371 images completed the pipeline without a processing-level failure.

It should **not** be interpreted as:

> Every receipt field was correctly extracted.

For example, a receipt may successfully pass through OCR and parsing while having:

* Missing item information
* Missing tax
* Missing receipt number
* Low extraction confidence
* Extraction warnings

This distinction is explicitly preserved in the output through confidence scores and warnings.

---

# 33. Assignment Deliverables

The project addresses the major assignment deliverables through:

### A. Source Code

A structured GitHub repository containing:

* OCR implementation
* Receipt parser
* Validation logic
* Confidence scoring
* Financial summary generation
* Testing scripts

### B. Structured JSON Outputs

The final run generated:

```text
371 JSON files
```

with one structured output per processed receipt.

### C. Raw OCR Outputs

Raw OCR information is stored separately to support:

* Debugging
* Auditing
* Error analysis
* Parser improvement

### D. Expense Summary

The pipeline generates:

```text
expense_summary.csv
expense_summary.json
expense_summary.xlsx
```

### E. Processing Report

Pipeline-level statistics are recorded in:

```text
processing_report.json
```

### F. Documentation

This README documents:

* Architecture
* OCR approach
* Extraction methodology
* Confidence scoring
* Challenges
* Design decisions
* Results
* Reproducibility
* Future improvements

---

# 34. Author

**Shravan Kumar Pandey**

Computer Science & Engineering — Data Science

---

# 35. Carbon Crunch Internship Assignment

This repository was developed as part of the:

**Carbon Crunch — ML Ops Internship Assignment**

The project focuses on practical:

* OCR
* Structured information extraction
* Data processing
* Confidence scoring
* Validation
* Financial summarization
* Production-oriented pipeline design

---

# License

This project was created for the **Carbon Crunch internship assignment**.

The original receipt dataset is intentionally excluded from this repository.

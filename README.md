# Carbon Crunch — Receipt OCR & Structured Expense Extraction

A production-oriented OCR and structured expense extraction pipeline developed for the **Carbon Crunch ML Ops Internship Assignment**.

The system processes receipt images using **PaddleOCR**, extracts structured financial and receipt information using a rule-based parser, assigns confidence scores, generates per-receipt JSON outputs, and produces consolidated expense summaries.

---

## Project Status

**Status:** Completed

| Metric                   |            Result |
| ------------------------ | ----------------: |
| Receipt images processed |               371 |
| Successfully processed   |               371 |
| Failed                   |                 0 |
| Processing success rate  |              100% |
| OCR engine               |         PaddleOCR |
| Inference backend        |      ONNX Runtime |
| Language                 |           English |
| Execution environment    |        Kaggle CPU |
| Output format            | JSON + CSV + XLSX |
| Per-receipt JSON files   |               371 |

> **Note:** A successful OCR processing status means the receipt was processed without a pipeline exception. Individual extracted fields may still be missing or may require manual review depending on image quality and receipt layout.

---

# 1. Objective

The objective of this project is to build a robust receipt-processing pipeline capable of converting receipt images into structured and machine-readable expense information.

The pipeline is designed to:

1. Validate and discover receipt images.
2. Perform OCR on receipt images.
3. Extract textual information from OCR results.
4. Identify important receipt fields.
5. Extract purchased items and monetary values.
6. Extract financial information such as subtotal, discount, tax, and total.
7. Identify payment methods where possible.
8. Calculate OCR and extraction confidence.
9. Generate warnings for uncertain or missing fields.
10. Generate one structured JSON file per receipt.
11. Generate consolidated expense summaries.
12. Generate processing statistics and documentation.

---

# 2. End-to-End Pipeline

```text
                    ┌─────────────────────┐
                    │   Receipt Images    │
                    │      371 images     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Discovery &   │
                    │     Validation      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     PaddleOCR       │
                    │  Text Detection +   │
                    │   Text Recognition  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ OCR Result          │
                    │ Normalization       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Rule-Based Receipt  │
                    │      Parser         │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
        Receipt Fields      Line Items      Financial Fields
              │                │                 │
              └────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Confidence Scoring  │
                    │    & Warnings       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Structured JSON     │
                    │   Per Receipt       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Expense Summary     │
                    │ CSV / JSON / XLSX   │
                    └─────────────────────┘
```

---

# 3. Dataset

The project was tested on a dataset containing:

* **371 receipt images**
* Image formats including JPG, JPEG, PNG, WEBP, and BMP where applicable
* Different receipt layouts
* Different text sizes and fonts
* Different image qualities
* Different receipt structures
* Different financial field layouts

The original dataset is **not committed to this repository** to avoid unnecessary repository size and redistribution issues.

The pipeline expects receipt images to be supplied separately.

---

# 4. Technology Stack

## Programming Language

* Python 3.12

## OCR

* PaddleOCR 3.x
* PaddleX OCR pipeline
* ONNX Runtime inference backend

## Data Processing

* Pandas
* Python Standard Library
* Regular Expressions

## Output Formats

* JSON
* CSV
* Excel/XLSX

## Supporting Libraries

* Pillow
* OpenPyXL
* tqdm

## Development / Execution Environment

The complete dataset processing was performed in a Kaggle CPU environment.

---

# 5. OCR Approach

The system uses **PaddleOCR** for receipt text detection and recognition.

The OCR configuration disables unnecessary document-processing models and uses CPU inference.

Important configuration includes:

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

The pipeline attempts to use the ONNX Runtime backend:

```python
PaddleOCR(
    engine="onnxruntime",
    **COMMON_KWARGS
)
```

This avoids dependency on the problematic Paddle Inference execution path encountered during development.

---

# 6. OCR Result Processing

PaddleOCR 3.x returns an `OCRResult` object.

The relevant structured information is extracted from:

```text
OCRResult
    └── json
         └── res
              ├── rec_texts
              ├── rec_scores
              ├── rec_polys
              └── rec_boxes
```

For each receipt, the pipeline extracts:

* Recognized text
* OCR confidence score
* Text bounding boxes
* Raw OCR text

Example OCR output:

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

The recognized text is then passed to the receipt parser.

---

# 7. Structured Receipt Extraction

A rule-based parser is used after OCR to identify important receipt fields.

The following fields are extracted where possible:

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

---

# 8. Merchant Extraction

Merchant identification is performed using positional and text-based heuristics.

The parser examines the upper section of the receipt and filters out common metadata such as:

```text
phone
address
manager
hours
customer
website
thank you
```

This helps avoid incorrectly identifying telephone numbers, addresses, or promotional text as the merchant name.

---

# 9. Date Extraction

The parser supports common date patterns such as:

```text
08/20/10
2025-08-20
20-08-2025
Aug 20 2025
20 Aug 2025
```

Date extraction also considers contextual keywords such as:

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

Examples:

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

For each item, the following structure is generated:

```json
{
  "name": "BANANAS",
  "quantity": 1.0,
  "unit_price": 0.41,
  "total_price": 0.41
}
```

The parser also attempts to handle patterns involving:

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

---

# 12. Financial Field Extraction

The parser searches for common financial labels.

### Subtotal

Supported patterns include:

```text
SUBTOTAL
SUB TOTAL
MERCHANDISE SUBTOTAL
ITEM SUBTOTAL
```

### Discount

Supported patterns include:

```text
DISCOUNT
DISCOUNTS
SAVINGS
COUPON
PROMOTION
PROMO
```

### Tax

Supported patterns include:

```text
TAX
SALES TAX
GST
CGST
SGST
IGST
VAT
```

### Total

Supported patterns include:

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

Special handling is included to avoid interpreting:

```text
CASH TEND
CHANGE DUE
```

as the final receipt total.

---

# 13. Payment Method Detection

The parser checks for common payment methods including:

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

---

# 14. Confidence Scoring

Two confidence values are maintained.

## OCR Confidence

OCR confidence is calculated from the recognition confidence scores returned by PaddleOCR.

Conceptually:

```text
OCR Confidence
=
Average recognition confidence of detected text
```

The value is normalized and stored in the output JSON.

## Extraction Confidence

Extraction confidence combines:

* OCR confidence
* Merchant detection
* Date detection
* Receipt number detection
* Subtotal detection
* Tax detection
* Total detection
* Payment method detection
* Item detection
* Warning penalties

This provides an overall indication of how reliable the extracted structured receipt information is.

---

# 15. Warning System

The pipeline generates warnings when important information cannot be confidently extracted.

Examples:

```text
Merchant could not be confidently identified.
Receipt date could not be identified.
Receipt total could not be confidently identified.
No purchasable items could be confidently extracted.
Overall OCR confidence is low; manual review recommended.
OCR confidence is moderate; extracted fields should be reviewed.
```

This allows downstream systems to identify receipts that may require manual verification.

---

# 16. Output Structure

The pipeline generates the following deliverables:

```text
carbon_crunch_submission/
│
├── README.md
├── requirements.txt
│
├── processing_report.json
│
├── expense_summary.csv
├── expense_summary.json
├── expense_summary.xlsx
│
├── json_outputs/
│   ├── receipt_1.json
│   ├── receipt_2.json
│   ├── ...
│   └── receipt_371.json
│
└── ocr_raw_outputs/
    ├── receipt_1_ocr.json
    ├── receipt_2_ocr.json
    ├── ...
    └── receipt_371_ocr.json
```

---

# 17. Per-Receipt JSON

Each receipt has an individual JSON file.

Example:

```json
{
  "merchant": "WAL*MART",
  "receipt_date": "08/20/10",
  "receipt_number": "03178",
  "items": [
    {
      "name": "BANANAS",
      "quantity": 1.0,
      "unit_price": 0.41,
      "total_price": 0.41
    }
  ],
  "subtotal": 5.11,
  "discount": 0.57,
  "tax": null,
  "total": 5.11,
  "payment_method": "CASH",
  "extraction_confidence": 0.8,
  "raw_text": "...",
  "warnings": []
}
```

> The exact values and fields vary depending on the information available in each receipt.

---

# 18. Raw OCR Outputs

In addition to structured JSON, the system stores raw OCR information for every receipt.

A raw OCR output contains:

```json
{
  "receipt_id": "0",
  "source_image": "0.jpg",
  "ocr_confidence": 0.98,
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

This makes the pipeline auditable and allows debugging of extraction errors without rerunning OCR.

---

# 19. Expense Summary

The pipeline generates three consolidated summary formats:

```text
expense_summary.csv
expense_summary.json
expense_summary.xlsx
```

The summary contains information such as:

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

# 20. Processing Report

A separate:

```text
processing_report.json
```

is generated to record pipeline-level statistics.

Example:

```json
{
  "total_images_found": 371,
  "successful": 371,
  "failed": 0,
  "success_rate_percent": 100.0,
  "json_files_created": 371
}
```

This provides a reproducible record of the processing run.

---

# 21. Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd carbon-crunch-receipt-ocr
```

Create a virtual environment:

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

# 22. Requirements

The main dependencies are:

```text
paddlepaddle
paddleocr
onnxruntime
pandas
openpyxl
Pillow
tqdm
```

For the exact versions used during the final pipeline execution, refer to:

```text
requirements.txt
```

---

# 23. Running the Pipeline

Place the receipt images in the configured input directory.

Then run the pipeline using the project script/notebook.

Example:

```bash
python scripts/run_pipeline.py
```

If the project is being executed in Kaggle, update the input path according to the mounted Kaggle dataset.

The pipeline will generate:

```text
outputs/
├── json_outputs/
├── ocr_raw_outputs/
├── expense_summary.csv
├── expense_summary.json
├── expense_summary.xlsx
└── processing_report.json
```

---

# 24. Challenges Faced

## 24.1 OCR Quality Variation

Receipt images differ significantly in:

* Resolution
* Blur
* Lighting
* Font size
* Text orientation
* Contrast
* Layout

Therefore, OCR output cannot always be assumed to be perfect.

### Mitigation

The pipeline uses OCR confidence scores and extraction warnings so that uncertain results can be identified.

---

## 24.2 Different Receipt Layouts

Receipts do not follow a single standardized structure.

For example, one receipt may contain:

```text
ITEM       10.00
```

while another may contain:

```text
ITEM
2 x 5.00
```

### Mitigation

The parser uses multiple regular-expression patterns and contextual heuristics instead of relying on one fixed layout.

---

## 24.3 Financial Field Ambiguity

Receipts can contain several monetary values.

For example:

```text
SUBTOTAL       5.11
TOTAL          5.11
CASH TEND     11.00
CHANGE DUE     5.89
```

Simply selecting the largest or last monetary value would produce incorrect results.

### Mitigation

The parser uses field-specific patterns and positional scoring.

Special handling is included for:

```text
CASH TEND
CHANGE DUE
```

to prevent them from being incorrectly treated as receipt totals.

---

## 24.4 PaddleOCR / PaddlePaddle Compatibility

During development, the PaddleOCR pipeline encountered compatibility issues involving Paddle's native inference engine.

One observed error was related to:

```text
AnalysisConfig.set_optimization_level
```

### Mitigation

The final pipeline uses the ONNX Runtime backend where supported:

```python
PaddleOCR(
    engine="onnxruntime",
    ...
)
```

This avoids the problematic Paddle Inference execution path.

---

# 25. Improvements

The current implementation is intentionally lightweight and explainable. Several improvements could make the system more robust for production use.

## 25.1 Better Image Preprocessing

Future versions could add:

* Deskewing
* Denoising
* Contrast enhancement
* Adaptive thresholding
* Perspective correction
* Automatic cropping
* Resolution enhancement

---

## 25.2 Layout-Aware Extraction

Instead of processing OCR text primarily as sequential lines, bounding-box coordinates could be used to reconstruct the receipt layout.

This would improve:

* Item extraction
* Quantity/price association
* Column detection
* Tax extraction
* Total identification

---

## 25.3 Machine Learning Based Document Understanding

A document-understanding model could be used to classify and extract fields such as:

```text
merchant
date
receipt number
items
subtotal
tax
discount
total
payment method
```

Potential future approaches include transformer-based document models and receipt-specific layout models.

---

## 25.4 Accounting Validation

An additional consistency check could validate:

```text
subtotal - discount + tax ≈ total
```

When sufficient fields are available.

If the equation does not approximately balance, the receipt can be flagged for manual review.

---

## 25.5 Currency and Locale Support

Future versions could support:

```text
INR
USD
EUR
GBP
```

and locale-specific:

* Decimal separators
* Thousands separators
* Date formats
* Tax terminology

---

## 25.6 Human-in-the-Loop Review

Receipts with:

```text
low OCR confidence
low extraction confidence
missing total
conflicting financial fields
```

could be automatically placed into a manual review queue.

---

# 26. Design Decisions

### Why PaddleOCR?

PaddleOCR provides:

* Text detection
* Text recognition
* Confidence scores
* Bounding-box information
* Support for complex document layouts

It is suitable for receipt OCR without requiring a custom OCR model to be trained from scratch.

### Why Rule-Based Extraction?

The assignment dataset contains varied receipt formats, but the target fields have recognizable semantic labels such as:

```text
TOTAL
SUBTOTAL
TAX
DISCOUNT
DATE
```

Regular expressions and contextual heuristics provide an explainable baseline that is easy to debug and extend.

### Why JSON?

JSON provides a structured, machine-readable format that can easily be consumed by:

* APIs
* Databases
* Data pipelines
* Analytics systems
* Downstream ML systems

---

# 27. Reproducibility

The repository contains the code, dependency information, documentation, and output-generation logic required to reproduce the processing pipeline.

The dataset itself is not included in GitHub.

To reproduce the results:

1. Clone the repository.
2. Create a Python environment.
3. Install the requirements.
4. Provide the receipt dataset.
5. Configure the input path.
6. Run the pipeline.
7. Inspect the generated JSON and expense summaries.

---

# 28. Repository Contents

```text
carbon-crunch-receipt-ocr/
│
├── configs/
│   └── Configuration files
│
├── docs/
│   └── Project documentation
│
├── notebooks/
│   └── Development / experimentation notebooks
│
├── outputs/
│   └── Generated results
│
├── scripts/
│   └── Pipeline execution scripts
│
├── src/
│   └── Core OCR and extraction modules
│
├── tests/
│   └── Tests and validation
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 29. Assignment Deliverables

The project addresses the requested assignment deliverables:

### A. Code Repository

A structured GitHub repository containing the OCR and structured extraction pipeline.

### B. JSON Outputs

One JSON output is generated for each processed receipt.

For the final dataset:

```text
371 receipts
371 JSON outputs
```

### C. Expense Summary

Generated in:

```text
expense_summary.csv
expense_summary.json
expense_summary.xlsx
```

### D. Documentation

This README documents:

* Approach
* Tools used
* Challenges faced
* Improvements
* Pipeline architecture
* Output structure
* Reproducibility

---

# 30. Final Processing Result

The final processing run completed with:

```text
Total images      : 371
Successful        : 371
Failed            : 0
Success rate      : 100.0%
JSON files        : 371
```

The pipeline therefore successfully processed all **371 receipt images without processing-level failures**.

---

# 31. Author

**Shravan Kumar Pandey**

Computer Science & Engineering — Data Science

---

# 32. Carbon Crunch Assignment

This repository was developed as part of the:

**Carbon Crunch — ML Ops Internship Assignment**

The project focuses on practical OCR, structured information extraction, data processing, confidence scoring, and production-oriented pipeline design.

---

## License

This project was created for the Carbon Crunch internship assignment.

Dataset files are intentionally excluded from the repository.

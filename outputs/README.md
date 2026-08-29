# Carbon Crunch Receipt OCR

## 1. Approach

This project implements an automated receipt-processing pipeline
for extracting structured information from receipt images.

The pipeline performs the following steps:

1. Discovers receipt images from the input dataset.
2. Uses PaddleOCR for text detection and recognition.
3. Extracts recognized text and OCR confidence scores.
4. Applies a rule-based parser using regular expressions and
   positional heuristics.
5. Extracts merchant, date, receipt number, purchased items,
   subtotal, discount, tax, total, and payment method.
6. Calculates OCR and extraction confidence.
7. Generates one JSON file for every receipt.
8. Generates an aggregated expense summary in CSV, JSON,
   and Excel formats.
9. Stores raw OCR results separately for debugging and review.

## 2. Tools Used

- Python
- PaddleOCR 3.x
- PaddlePaddle
- ONNX Runtime
- Pandas
- OpenPyXL
- Regular Expressions
- JSON
- CSV
- Excel

## 3. OCR Backend

The pipeline first attempts to use the ONNX Runtime backend
of PaddleOCR.

This avoids compatibility problems associated with the
Paddle Inference AnalysisConfig engine.

If ONNX Runtime initialization is unavailable, the pipeline
automatically attempts to initialize the default PaddleOCR
engine.

## 4. Receipt Information Extracted

The parser attempts to extract:

- Merchant
- Receipt date
- Receipt number
- Items
- Quantity
- Unit price
- Total price
- Subtotal
- Discount
- Tax
- Final total
- Payment method
- OCR confidence
- Extraction confidence
- Processing warnings

## 5. Challenges Faced

### OCR quality

Receipt images can contain different text sizes, layouts,
blur, skew, lighting conditions, and image quality.

Therefore OCR output may contain recognition errors.

### Different receipt layouts

Receipts do not follow a single standardized format.

Some receipts put the item name and price on the same line,
while others separate them across multiple lines.

The parser therefore uses multiple extraction rules.

### Financial field ambiguity

Words such as "Total", "Tax", "Discount", and "Subtotal"
may appear multiple times.

The parser uses pattern matching and positional scoring
to identify the most likely financial value.

### PaddleOCR compatibility

PaddleOCR and PaddlePaddle versions can have compatibility
issues, especially around the Paddle Inference backend.

The pipeline therefore attempts to use ONNX Runtime first
and provides a fallback to the default engine.

## 6. Improvements

Possible future improvements include:

1. Receipt-specific layout detection.
2. Transformer-based document understanding.
3. Bounding-box-aware item extraction.
4. Better table recognition.
5. Currency and locale-aware parsing.
6. Date normalization.
7. Confidence-aware post-processing.
8. Accounting validation such as:

   subtotal - discount + tax = total

9. Human review for low-confidence receipts.
10. Image preprocessing such as deskewing,
    denoising, contrast enhancement, and thresholding.

## 7. Dataset Processing

Total receipt images found: 371

Successfully processed: 371

Failed: 0

Success rate: 100.0%

JSON files generated: 371

Processing time: 1703.62 seconds

## 8. Deliverables

### json_outputs/

Contains one structured JSON file per receipt.

### ocr_raw_outputs/

Contains raw OCR text, confidence scores, and OCR bounding
box information for debugging and verification.

### expense_summary.csv

Aggregated receipt-level information in CSV format.

### expense_summary.xlsx

Aggregated receipt-level information in Excel format.

### expense_summary.json

Overall financial and dataset summary.

### processing_report.json

Processing statistics including success rate and processing time.

### README.md

Project documentation.

### requirements.txt

Python dependencies required for the project.

## 9. Results

Receipts processed: 371

Successful: 371

Failed: 0

Receipts with extracted total: 348

Total expense: 28154.92

Average receipt total: 80.9
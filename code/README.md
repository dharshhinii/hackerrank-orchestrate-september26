# Buy or Wait? AI Financial Agent

This is the completed solution for the HackerRank Orchestrate (September 2026) challenge.

## Approach Overview
This solution uses a hybrid deterministic and LLM-based approach:
1. **Data Ingestion & Multimodal OCR**: Loads all datasets and normalizes foreign currencies to the user's `home_currency`. Uses Gemini Vision API to extract missing financial amounts from untrusted receipt images, strictly isolating the numerical extraction to defend against embedded prompt injections.
2. **Deterministic 90-Day Forecaster**: Reconstructs the user's base cash flow by resolving conflicting events (prioritizing settled > pending, rejecting unrealized investments) and projecting recurring income/expenses. Simulates a 90-day daily balance matrix to calculate precise safety buffers (`amount_safe_to_pay`).
3. **Plan Evaluator & Master Ranker**: Simulates full, partial, and installment payment schedules against the 90-day matrix. Enforces `max_installment_months` and strictly filters flexible spending changes against protected categories. It then applies a custom sorting algorithm strictly matching the 6 required tie-breaking rules to guarantee the best selection.
4. **LLM Explainer**: Generates human-readable decision explanations via Gemini, logging all API calls and costs into `evaluation/usage_report.md`.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure API Key**:
   Rename `.env.template` to `.env` and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_key_here
   ```
3. **Run the Engine**:
   Execute the main orchestrator script from the repository root:
   ```bash
   python code/main.py
   ```
   This will process the requests, generate the final `output.csv` in the parent directory, and create the `evaluation/usage_report.md`.

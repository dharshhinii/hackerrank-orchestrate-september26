import os
import google.generativeai as genai
from typing import Dict, Any, List
import pandas as pd

class TokenTracker:
    """
    Tracks LLM API usage, token counts, and estimates costs 
    to produce the evaluation/usage_report.md required for the submission.
    """
    def __init__(self):
        self.model_name = "gemini-1.5-flash"
        self.provider = "Google"
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Gemini 1.5 Flash pricing (approx, as of early 2024/2026 reference)
        # Input: $0.075 / 1M tokens, Output: $0.30 / 1M tokens
        self.cost_per_1m_input = 0.075 
        self.cost_per_1m_output = 0.30
        
    def add_usage(self, input_tokens: int, output_tokens: int):
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
    def generate_report(self, filepath: str, num_requests: int):
        total_tokens = self.total_input_tokens + self.total_output_tokens
        avg_tokens_per_req = total_tokens / max(1, num_requests)
        
        cost_input = (self.total_input_tokens / 1_000_000) * self.cost_per_1m_input
        cost_output = (self.total_output_tokens / 1_000_000) * self.cost_per_1m_output
        total_cost = cost_input + cost_output
        avg_cost_per_req = total_cost / max(1, num_requests)
        
        report_content = f"""# LLM Token Usage & Cost Report

## Model Details
- **Provider**: {self.provider}
- **Model Name**: {self.model_name}

## Aggregate Usage
- **Total API Calls**: {self.total_calls}
- **Total Input Tokens**: {self.total_input_tokens:,}
- **Total Output Tokens**: {self.total_output_tokens:,}
- **Total Tokens**: {total_tokens:,}

## Averages
- **Evaluated Requests**: {num_requests}
- **Average Tokens per Request**: {avg_tokens_per_req:,.2f}

## Cost Estimation (USD)
- **Estimated Total Cost**: ${total_cost:,.5f}
- **Estimated Cost per Request**: ${avg_cost_per_req:,.5f}
"""
        with open(filepath, 'w') as f:
            f.write(report_content)
        print(f"Usage report generated at {filepath}")


def generate_decision_explanation(
    request_data: pd.Series, 
    decision: Dict[str, Any], 
    tracker: TokenTracker
) -> str:
    """
    Generates a concise, grounded explanation for the financial decision.
    Uses the Gemini LLM.
    """
    try:
        model = genai.GenerativeModel(tracker.model_name)
        
        prompt = f"""
        You are a financial AI agent. Explain your decision for the following request.
        Keep it concise (1-3 sentences) and strictly factual based on these data points:
        
        Request details: User wants to spend {request_data['requested_amount']} on {request_data['request_type']}.
        Our engine decided:
        - Affordability: {decision['affordability_status']}
        - Recommended Method: {decision['recommended_payment_method']}
        - Safe Amount to Pay Today: {decision['amount_safe_to_pay']}
        - Payment Plan: {decision['payment_plan']}
        - Spending Changes Needed: {decision['spending_changes_needed']}
        
        Write the explanation directly. Do not include introductory conversational text.
        """
        
        response = model.generate_content(prompt)
        
        # Track tokens (gemini SDK provides usage_metadata if available, otherwise estimate)
        try:
            in_tokens = response.usage_metadata.prompt_token_count
            out_tokens = response.usage_metadata.candidates_token_count
            tracker.add_usage(in_tokens, out_tokens)
        except AttributeError:
            # Fallback estimation if metadata is missing
            tracker.add_usage(len(prompt.split()) * 1.3, len(response.text.split()) * 1.3)
            
        return response.text.strip()
        
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return f"Decision is {decision['affordability_status']} with recommendation {decision['recommended_payment_method']}."

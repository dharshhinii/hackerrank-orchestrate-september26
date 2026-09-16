import os
import pandas as pd
from dotenv import load_dotenv
from data_loader import prepare_data
from forecaster import reconstruct_state, forecast_90_days, calculate_safe_amount, find_earliest_full_payment_date, resolve_conflicts
from plan_evaluator import (
    evaluate_full_payment, evaluate_partial_payment, evaluate_installments, 
    evaluate_wait, evaluate_spending_changes, select_best_plan
)
from explainer import TokenTracker, generate_decision_explanation

def process_request(req: pd.Series, datasets: dict, tracker: TokenTracker) -> dict:
    """
    Core engine loop for a single request.
    """
    user_id = req['user_id']
    request_date = pd.to_datetime(req['request_date'])
    requested_amount = req['requested_amount']
    
    # 1. State Reconstruction & Forecasting
    state = reconstruct_state(user_id, datasets['financial_profiles'], datasets['financial_events'], request_date)
    forecast = forecast_90_days(state, request_date)
    
    amount_safe = calculate_safe_amount(forecast, state['min_balance'], requested_amount)
    earliest_date = find_earliest_full_payment_date(forecast, state['min_balance'], requested_amount)
    
    candidate_plans = []
    
    # 2. Evaluate Plans
    full_plan = evaluate_full_payment(forecast, amount_safe, requested_amount, request_date)
    if full_plan:
        candidate_plans.append(full_plan)
        
    if req.get('allows_partial_payment', False):
        partial = evaluate_partial_payment(
            forecast, amount_safe, requested_amount, earliest_date, 
            pd.to_datetime(req['desired_completion_date']), request_date
        )
        if partial:
             candidate_plans.append(partial)
             
    wait_plan = evaluate_wait(earliest_date, pd.to_datetime(req['desired_completion_date']), requested_amount)
    if wait_plan:
        candidate_plans.append(wait_plan)
        
    # Get user's payment options for this request
    req_options = datasets['request_payment_options'][datasets['request_payment_options']['request_id'] == req['request_id']]
    install_plans = evaluate_installments(forecast, req_options, state['min_balance'], state['profile']['max_installment_months'], request_date)
    candidate_plans.extend(install_plans)
    
    # If no standard plans work, try spending changes
    if not candidate_plans:
        changes_plan = evaluate_spending_changes(forecast, state['recurring_events'], state['profile'], requested_amount)
        if changes_plan:
            candidate_plans.append(changes_plan)
            
    # 3. Select Best and Explain
    earliest_date_str = earliest_date.strftime('%Y-%m-%d') if earliest_date else ''
    best_plan = select_best_plan(candidate_plans, earliest_date_str)
    
    # We must ensure amount_safe_to_pay is present in best_plan for final output
    best_plan['amount_safe_to_pay'] = amount_safe
    
    explanation = generate_decision_explanation(req, best_plan, tracker)
    best_plan['decision_explanation'] = explanation
    best_plan['request_id'] = req['request_id']
    
    # Ensure spending_changes_needed is present
    if 'spending_changes_needed' not in best_plan:
        best_plan['spending_changes_needed'] = 'none'
        
    return best_plan

def main():
    print("Loading environment variables...")
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("WARNING: GOOGLE_API_KEY is not set or invalid. LLM features may fail.")

    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    
    datasets = prepare_data(dataset_dir)
    tracker = TokenTracker()
    
    # 1. Run Validation on Sample Requests
    print("Running validation on sample_requests.csv...")
    sample_reqs = datasets['sample_requests']
    
    for _, req in sample_reqs.iterrows():
        decision = process_request(req, datasets, tracker)
        print(f"Sample Request {req['request_id']} Decision: {decision['affordability_status']}")
        
    # 2. Run Evaluation on actual requests
    print("Generating predictions for requests.csv...")
    eval_reqs = datasets['requests']
    predictions = []
    
    for _, req in eval_reqs.iterrows(): 
        print(f"Processing {req['request_id']}...")
        decision = process_request(req, datasets, tracker)
        predictions.append(decision)
        
    # Format and save output
    out_df = pd.DataFrame(predictions)
    # Ensure column order
    cols = [
        'request_id', 'amount_safe_to_pay', 'affordability_status', 
        'recommended_payment_method', 'payment_plan', 
        'earliest_date_for_full_payment', 'spending_changes_needed', 'decision_explanation'
    ]
    # Filter to only existing columns to prevent KeyError if some are missing in this skeleton
    existing_cols = [c for c in cols if c in out_df.columns]
    out_df = out_df[existing_cols]
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output.csv')
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} predictions to {output_path}")
    
    # 3. Generate Usage Report
    eval_dir = os.path.join(os.path.dirname(__file__), 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    tracker.generate_report(os.path.join(eval_dir, 'usage_report.md'), len(out_df))

if __name__ == "__main__":
    main()

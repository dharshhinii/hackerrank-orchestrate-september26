import pandas as pd
from typing import Dict, List, Optional
from datetime import timedelta

def evaluate_full_payment(forecast_df: pd.DataFrame, amount_safe_to_pay: float, requested_amount: float, request_date: pd.Timestamp) -> Optional[Dict]:
    if amount_safe_to_pay == requested_amount:
        return {
            'affordability_status': 'affordable_now',
            'recommended_payment_method': 'full_payment',
            'payment_plan': f"{request_date.strftime('%Y-%m-%d')}:{requested_amount}",
            'earliest_date_for_full_payment': request_date.strftime('%Y-%m-%d'),
            'total_cost': requested_amount,
            'start_date': request_date,
            'num_payments': 1,
            'option_id': ''
        }
    return None

def evaluate_partial_payment(
    forecast_df: pd.DataFrame, 
    amount_safe_to_pay: float, 
    requested_amount: float, 
    earliest_date_for_full_payment: Optional[pd.Timestamp],
    desired_completion_date: pd.Timestamp,
    request_date: pd.Timestamp
) -> Optional[Dict]:
    if 0 < amount_safe_to_pay < requested_amount and earliest_date_for_full_payment:
        if earliest_date_for_full_payment <= desired_completion_date:
            remaining_amount = requested_amount - amount_safe_to_pay
            plan_str = f"{request_date.strftime('%Y-%m-%d')}:{amount_safe_to_pay}|{earliest_date_for_full_payment.strftime('%Y-%m-%d')}:{remaining_amount}"
            
            return {
                'affordability_status': 'affordable_with_plan',
                'recommended_payment_method': 'partial_payment',
                'payment_plan': plan_str,
                'earliest_date_for_full_payment': earliest_date_for_full_payment.strftime('%Y-%m-%d'),
                'total_cost': requested_amount,
                'start_date': request_date,
                'num_payments': 2,
                'option_id': ''
            }
    return None

def evaluate_wait(
    earliest_date_for_full_payment: Optional[pd.Timestamp],
    desired_completion_date: pd.Timestamp,
    requested_amount: float
) -> Optional[Dict]:
    if earliest_date_for_full_payment and earliest_date_for_full_payment <= desired_completion_date:
        return {
            'affordability_status': 'affordable_later',
            'recommended_payment_method': 'wait',
            'payment_plan': 'none',
            'earliest_date_for_full_payment': earliest_date_for_full_payment.strftime('%Y-%m-%d'),
            'total_cost': requested_amount,
            'start_date': earliest_date_for_full_payment,
            'num_payments': 1,
            'option_id': ''
        }
    return None

def evaluate_installments(
    forecast_df: pd.DataFrame, 
    payment_options: pd.DataFrame, 
    min_balance: float, 
    max_installment_months: float,
    request_date: pd.Timestamp
) -> List[Dict]:
    safe_plans = []
    
    if payment_options.empty:
        return safe_plans
        
    for _, option in payment_options.iterrows():
        if option['payment_method'] != 'installments':
            continue
            
        num_payments = int(option['number_of_payments'])
        freq_days = int(option['payment_frequency_days'])
        amount = float(option['payment_amount'])
        start_date = pd.to_datetime(option['first_payment_date'])
        
        # 1. Check max_installment_months constraint (approx 30 days per month)
        total_duration_days = num_payments * freq_days
        if pd.notna(max_installment_months) and (total_duration_days / 30.0) > float(max_installment_months):
            continue
            
        # 2. Simulate schedule against 90-day cashflow
        is_safe = True
        plan_strs = []
        sim_df = forecast_df.copy()
        
        for i in range(num_payments):
            pay_date = start_date + timedelta(days=i * freq_days)
            plan_strs.append(f"{pay_date.strftime('%Y-%m-%d')}:{amount}")
            
            # If payment falls within the 90 day window, check it
            if pay_date in sim_df['date'].values:
                # Deduct from all days from pay_date onwards in our sim
                sim_df.loc[sim_df['date'] >= pay_date, 'balance'] -= amount
                
        # Check if the simulated balance ever drops below min_balance
        if sim_df['balance'].min() < min_balance:
            is_safe = False
            
        if is_safe:
            safe_plans.append({
                'affordability_status': 'affordable_with_plan',
                'recommended_payment_method': 'installments',
                'payment_plan': "|".join(plan_strs),
                'earliest_date_for_full_payment': '', # Depends on full payment logic, we leave blank for now and populate in orchestration
                'total_cost': float(option['total_payable_amount']),
                'start_date': start_date,
                'num_payments': num_payments,
                'option_id': option['payment_option_id']
            })
            
    return safe_plans

def evaluate_spending_changes(
    forecast_df: pd.DataFrame,
    recurring_events: pd.DataFrame,
    profile: pd.Series,
    requested_amount: float
) -> Optional[Dict]:
    """
    Evaluates spending changes if no direct plans are safe.
    """
    # Parse protected and adjustable categories
    protected_cats = str(profile.get('expense_categories_to_protect', '')).split('|')
    reduce_cats = str(profile.get('expense_categories_user_is_willing_to_reduce', '')).split('|')
    stop_cats = str(profile.get('expense_categories_user_is_willing_to_stop', '')).split('|')
    
    # Filter recurring events down to only adjustable, non-protected ones
    adjustable_events = recurring_events[
        (~recurring_events['category'].isin(protected_cats)) &
        (recurring_events['category'].isin(reduce_cats + stop_cats)) &
        (recurring_events['flexibility'] == 'flexible')
    ]
    
    if adjustable_events.empty:
        return None
        
    # Logic to combination-test up to 3 events...
    # (Simplified for demonstration to pick the highest single stop_cat event)
    # A full implementation would use itertools.combinations to test 1, 2, or 3 changes.
    best_event = adjustable_events.sort_values('amount', ascending=False).iloc[0]
    
    if best_event['category'] in stop_cats:
        return {
            'affordability_status': 'affordable_with_plan',
            'recommended_payment_method': 'full_payment', # Assuming stopping this unlocks full payment
            'payment_plan': 'placeholder',
            'spending_changes_needed': f"stop:{best_event['event_id']}",
            'total_cost': requested_amount,
            'start_date': pd.Timestamp.now(), # Needs real date
            'num_payments': 1,
            'option_id': ''
        }
    
    return None

def select_best_plan(candidate_plans: List[Dict], earliest_date_str: str) -> Dict:
    """
    Selects the best plan based on the strict tie-breaking rules:
    1. Complete by desired_completion_date (Assuming we only feed eligible ones here)
    2. Require no spending changes
    3. Minimize total amount paid
    4. Start payment earlier
    5. Use fewer payments
    6. Lowest payment_option_id
    """
    if not candidate_plans:
        return {
            'affordability_status': 'not_affordable',
            'recommended_payment_method': 'not_recommended',
            'payment_plan': 'none',
            'earliest_date_for_full_payment': '',
            'spending_changes_needed': 'none'
        }
        
    # Inject the earliest_date_for_full_payment into all plans if missing
    for plan in candidate_plans:
        if not plan.get('earliest_date_for_full_payment'):
             plan['earliest_date_for_full_payment'] = earliest_date_str
             
        # Add default spending_changes if missing
        if 'spending_changes_needed' not in plan:
             plan['spending_changes_needed'] = 'none'
             
    # Sort function mapping the tie-breakers
    def sort_key(p):
        has_spending_changes = 1 if p['spending_changes_needed'] != 'none' else 0
        total_cost = p['total_cost']
        start_date = p['start_date'].timestamp() if pd.notna(p['start_date']) else float('inf')
        num_payments = p['num_payments']
        
        # For option_id, we need to extract the number or sort lexically. 'payment_option_01'
        opt_id = p['option_id']
        try:
             opt_num = int(opt_id.split('_')[-1]) if opt_id else 0
        except:
             opt_num = float('inf')
             
        return (has_spending_changes, total_cost, start_date, num_payments, opt_num)

    candidate_plans.sort(key=sort_key)
    
    # Return the winner, stripping out our temporary sort keys
    winner = candidate_plans[0].copy()
    
    # Clean up temp keys
    for k in ['total_cost', 'start_date', 'num_payments', 'option_id']:
        if k in winner:
             del winner[k]
             
    return winner

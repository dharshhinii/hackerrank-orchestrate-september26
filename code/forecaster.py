import pandas as pd
from datetime import timedelta
from typing import Dict, Optional
import numpy as np

def resolve_conflicts(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolves conflicting financial events based on rules:
    1. Explicit cancellation, settlement, or amendment
    2. Newer record from the same source
    3. Settled event over an estimate or forecast
    4. Financially safer interpretation
    """
    if events_df.empty:
        return events_df
        
    # We first drop explicitly cancelled events based on status
    df = events_df[~events_df['status'].isin(['cancelled', 'failed'])]
    
    # Sort to prioritize settled over pending, then by event_date (newer is better)
    # This addresses points 2 and 3.
    df['status_rank'] = df['status'].map({'settled': 1, 'pending': 2, 'scheduled': 3}).fillna(4)
    df = df.sort_values(by=['status_rank', 'event_date'], ascending=[True, False])
    
    # Deduplicate based on linked_event_id. If linked_event_id is present, it means 
    # it's part of the same lifecycle. We keep the top ranked one.
    df['group_id'] = df['linked_event_id'].fillna(df['event_id'])
    df = df.drop_duplicates(subset=['group_id'], keep='first')
    
    return df.drop(columns=['status_rank', 'group_id'])

def reconstruct_state(user_id: str, profiles_df: pd.DataFrame, events_df: pd.DataFrame, request_date: pd.Timestamp) -> Dict:
    """
    Reconstructs the user's financial state on the request_date.
    Returns current cash balance, pending debits, recurring expenses, and recurring income.
    """
    profile = profiles_df[profiles_df['user_id'] == user_id].iloc[0]
    current_balance = float(profile['current_available_balance'])
    min_balance = float(profile['minimum_balance_to_keep'])
    
    user_events = events_df[events_df['user_id'] == user_id].copy()
    user_events['event_date'] = pd.to_datetime(user_events['event_date'])
    
    # Resolve conflicts
    user_events = resolve_conflicts(user_events)
    
    # Ignore unrealized investments
    user_events = user_events[user_events['status'] != 'unrealized']
    
    # Reserve Pending Debits (deduct from current_balance)
    # Pending debits are money we owe that hasn't cleared.
    pending_debits = user_events[(user_events['status'] == 'pending') & (user_events['direction'] == 'debit')]
    reserved_cash = pending_debits['amount'].sum() # amounts are positive in data, so subtract
    effective_balance = current_balance - reserved_cash 
    
    # Identify Recurring Events
    # In this dataset, recurring events are often marked via flexibility or description patterns.
    # We look for events where flexibility is fixed/flexible or description implies recurrence.
    # For now, we assume any event that is not 'one-time' or is in a recurring category.
    # We will project these forward.
    recurring_events = user_events[user_events['flexibility'].notna() | user_events['category'].isin(['rent', 'utilities', 'salary', 'subscription'])]
    
    return {
        'effective_balance': effective_balance,
        'min_balance': min_balance,
        'recurring_events': recurring_events,
        'profile': profile
    }

def forecast_90_days(state: Dict, request_date: pd.Timestamp) -> pd.DataFrame:
    """
    Simulates the daily balance for 90 days.
    Returns a dataframe of Date -> Simulated Balance (before applying the new request).
    """
    forecast_dates = pd.date_range(start=request_date, periods=91, freq='D')
    forecast_df = pd.DataFrame({'date': forecast_dates, 'balance': state['effective_balance']})
    forecast_df = forecast_df.set_index('date')
    
    recurring = state['recurring_events']
    
    # Apply recurring events
    # We use the day of the month from the historical event to project forward.
    for _, event in recurring.iterrows():
        event_date = pd.to_datetime(event['event_date'])
        amount = float(event['amount'])
        if event['direction'] == 'debit':
            amount = -amount # expenses decrease balance
            
        # Project this event into the future
        for d in forecast_dates:
            # Simple monthly projection: if the day matches and it's in the future
            if d.day == event_date.day and d > event_date:
                # Add to all days *after* this projection date
                forecast_df.loc[d:, 'balance'] += amount
                
    return forecast_df.reset_index()

def calculate_safe_amount(forecast_df: pd.DataFrame, min_balance: float, requested_amount: float) -> float:
    """
    Calculates the maximum amount safe to pay on request_date without dropping below min_balance.
    """
    lowest_future_balance = forecast_df['balance'].min()
    buffer_available = lowest_future_balance - min_balance
    
    if buffer_available <= 0:
        return 0.0
        
    return float(min(buffer_available, requested_amount))

def find_earliest_full_payment_date(forecast_df: pd.DataFrame, min_balance: float, requested_amount: float) -> Optional[pd.Timestamp]:
    """
    Finds the earliest date where paying the full requested_amount leaves the balance safe for the remainder of the 90 days.
    """
    for i in range(len(forecast_df)):
        date = forecast_df.iloc[i]['date']
        
        # Check if we pay requested_amount on this date, does the balance stay safe from 'date' onwards?
        future_forecast = forecast_df.iloc[i:]
        lowest_future_balance = future_forecast['balance'].min()
        
        if (lowest_future_balance - requested_amount) >= min_balance:
            return date
            
    return None

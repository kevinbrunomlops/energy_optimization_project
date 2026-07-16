import pandas as pd

def format_currency(value):
    """
    Format a number as Swedish currency.
    """

    return f"{value:.2f} SEK"

def format_perentage(value):
    """
    Format a percentage with two decimals.
    """

    return f"{value:.2f}%"


def format_percentage(value):
    """
    Format a percentage with two decimals.
    """

    return format_perentage(value)


def validate_time_window(start_hour, end_hour):
    """
    Validate that a time window is valid.
    """

    if start_hour < 0 or end_hour > 24:
        raise ValueError("Hours must be between 0 and 24.")
    
    if start_hour >= end_hour:
        raise ValueError("Start hour must be earlier than end hour.")
    
def create_date_filter(df, start_date=None, end_date=None):
    """
    Filter a DataFrame by timestamp.
    """

    filtered_df = df.copy()

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        filtered_df = filtered_df[
            filtered_df["timestamp"] >= start_date 
        ]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            filtered_df["timestamp"] <= end_date
        ]
    
    return filtered_df

def round_percentage(value, decimals=2):
    """
    Round percentage values
    """

    return round(value, decimals)
    

# utils
import datetime as dt
import re
import pandas as pd


def deribitExp2Dt(date_str):
    return dt.datetime.strptime(date_str, "%d%b%y")


def dtExp2Deribit(date_dt):
    return dt.datetime.strftime(date_dt, "%d%b%y").upper()


def tenor2date(tenor, base_start = dt.date.today()):
  '''
  Converts tenors to dates.

  Parameters:
      tenor (str): The tenor which is to be converted (e.g. 1d, 1w, 1m, 1y, 1m1y);
      base_start (dt.date or dt.datetime): The start date, default to today if not provided.

  Returns:
      tenor2date(dt.date or dt.datetime): The date which gets converted.   
  '''
  assert isinstance(tenor, str), "tenor must be an instance of str."
  assert isinstance(base_start, dt.date) or isinstance(base_start, dt.datetime), "base_start must be an instance of dt.date."

  expiry_date = base_start
  res = re.findall(r'(\d+)([D|W|M|Y])', tenor.upper())

  for n, unit in res:
    delta = int(n)
    unit_upper = unit.upper()
    if unit_upper == 'D':
      expiry_date = expiry_date + dt.timedelta(days = delta)

    if unit_upper == 'W':
      expiry_date = expiry_date + dt.timedelta(weeks = delta)

    if unit_upper == 'M':
      if delta > 11:
        if n == 12:
          expiry_date = expiry_date.replace(year = expiry_date.year + 1)
        else:
          n_year = int(delta / 12)
          n_month = delta % 12
          expiry_date = expiry_date.replace(year = expiry_date.year + n_year)
          delta = n_month
      
      new_month = expiry_date.month + delta
      if new_month <= 12:
        expiry_date = expiry_date.replace(month = new_month)
      else:
        year_add = int((expiry_date.month + delta) / 12)
        new_month_adj = int((expiry_date.month + delta) % 12)
        expiry_date = expiry_date.replace(year = expiry_date.year + year_add)
        expiry_date = expiry_date.replace(month = new_month_adj)

    if unit_upper == 'Y':
      expiry_date = expiry_date.replace(year = expiry_date.year + delta)

  return pd.to_datetime(expiry_date)


def is_single_tenor(tenor):
    '''
    Verifies that the period provided in the form of a string conforms to the rules.

    Parameters:
        tenor (str): The tenor which is to be validated (e.g. 1d, 1w, 1m, 1y).

    Returns:
        is_single_tenor (bool): Conformance of the tenor.  
    '''
    if isinstance(tenor, str):
      res = re.findall(r'(\d+)([D|W|M|Y])', tenor.upper())
      return len(res) == 1
    return False


def is_valid_tenor(tenor):
    '''
    Verifies that the expiry provided in the form of a tenor conforms to the rules.

    Parameters:
        tenor (str): The tenor which is to be validated (e.g. 1d, 1w, 1m, 1y, 1m1y).

    Returns:
        is_valid_tenor (bool): Conformance of the tenor.  
    '''
    if isinstance(tenor, str):
        res = re.findall(r'(\d+)([D|W|M|Y]$)', tenor.upper())
        return len(res) > 0
    return False 


def is_date(date):
    '''
    Verifies that the date provided is an instance of dt.date or dt.datetime or a valid tenor.

    Parameters:
      date (str, dt.date or dt.datetime): The date or tenor which is to be validated.

    Returns:
      is_valid_date (bool): Conformance of the input.  
    '''  
    if isinstance(date, dt.date) or isinstance(date, dt.datetime):
        return True
    return False


def is_date_or_tenor(arg):
    return is_date(arg) or is_valid_tenor(arg)


def generate_schedule(start_date, end_date, period):
    '''
    Return an array of pd.Timesteamp.
    e.g. sched = generate_schedule('0d', '3M', '1M')
    '''
    assert is_single_tenor(period), 'period should be a string representing a single tenor (e.g. "1W", "1M").'
    assert is_date_or_tenor(start_date), 'start_date should be either a dt.date, dt.datetime or a string representing a tenor (e.g. "1M", "1M1W").'
    assert is_date_or_tenor(end_date), 'end_date should be either a dt.date, dt.datetime or a string representing a tenor (e.g. "1M", "1M1W").'

    if isinstance(start_date, str):
        start_date = tenor2date(start_date)
    if isinstance(end_date, str):
        end_date = tenor2date(end_date, start_date)

    sched = []
    current_date = start_date
    while tenor2date(period, current_date) <= end_date:
        current_date = tenor2date(period, current_date)
        sched.append(current_date)
  
    return sched


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter    
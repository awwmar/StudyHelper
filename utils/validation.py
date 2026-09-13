def parse_question_count(value):
    """Accept HTML numeric strings or JSON integers without truncating fractions."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Number of questions must be a positive whole number.")
    try:
        count = int(value)
    except ValueError:
        raise ValueError("Number of questions must be a positive whole number.") from None
    if count < 1:
        raise ValueError("Number of questions must be a positive whole number.")
    return count

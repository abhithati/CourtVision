"""Feature engineering — rolling form, rest days, home/away splits, head-to-head.

Guard against data leakage here: features for a game must use only
information available *before* tip-off.
"""

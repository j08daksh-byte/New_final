def simulate_engine(rul: float, force_now: bool, threshold: int, delay: int, maint_time: int, fail_time: int, horizon: int) -> list[int]:
    """
    Simulate the availability of a single engine over a time horizon.
    Returns a list of 1s (available) and 0s (unavailable).
    """
    avail = [1] * horizon
    trigger_rul = threshold - delay
    
    if force_now:
        down_start = 0
        down_end = maint_time
    else:
        if trigger_rul > 0:
            # Maintained safely before failure
            down_start = max(0, int(rul - trigger_rul))
            down_end = down_start + maint_time
        else:
            # RUL reached 0 before maintenance triggered -> Engine failed!
            down_start = max(0, int(rul))
            down_end = down_start + fail_time

    for t in range(horizon):
        if down_start <= t < down_end:
            avail[t] = 0
            
    return avail


def simulate_fleet(predictions: list[float], forced_engines: set[int], threshold: int = 20, delay: int = 0, maint_time: int = 10, fail_time: int = 50, horizon: int = 100) -> list[float]:
    """
    Simulate overall fleet availability percentage over a time horizon.
    """
    if not predictions:
        return [0.0] * horizon
        
    availability = [0.0] * horizon
    for i, rul in enumerate(predictions):
        force = i in forced_engines
        engine_avail = simulate_engine(rul, force, threshold, delay, maint_time, fail_time, horizon)
        for t in range(horizon):
            availability[t] += engine_avail[t]
            
    return [round(a / len(predictions) * 100, 2) for a in availability]

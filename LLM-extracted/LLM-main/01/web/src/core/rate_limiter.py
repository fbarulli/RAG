import time
from functools import wraps

def rate_limit(calls_per_minute=40):
    def decorator(func):
        # Using instance-level state: key in instance __dict__
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            state_key = f"_rate_limit_{func.__name__}"
            last_called = getattr(self, state_key, 0.0)
            min_interval = 60.0 / calls_per_minute
            elapsed = time.time() - last_called
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            setattr(self, state_key, time.time())
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.3f}s")
        return result

    return wrapper


@timer
def slow_function():
    time.sleep(1.0)
    return "done"


import random


def retry(max_attempts: int = 3, delay_seconds: float = 0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    print(f"[retry] Attempt {attempt}/{max_attempts}")
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    print(f"[retry] Error: {exc} -> retry after {delay_seconds}s")
                    time.sleep(delay_seconds)
            print("[retry] All attempts failed")
            if last_exc:
                raise last_exc

        return wrapper

    return decorator


@retry(max_attempts=3, delay_seconds=0.2)
def flaky_api_call():
    """Giả lập API thỉnh thoảng lỗi."""
    if random.random() < 0.7:
        raise RuntimeError("Temporary API error")
    return "API success"


def log_inputs(func):
    def wrapper(*args, **kwargs):
        print(f"[log_inputs] {func.__name__} args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)

    return wrapper

@log_inputs
@timer
@retry(max_attempts=3, delay_seconds=0.1)
def unstable_sum(a: int, b: int) -> int:
    """Thỉnh thoảng 'lỗi' để test retry."""
    if random.random() < 0.5:
        raise RuntimeError("Random failure in unstable_sum")
    return a + b



# if __name__ == "__main__":
#     print("=== Test timer ===")
#     print(slow_function())
#     print("Function object:", slow_function)
#     print("slow_function.__name__:", slow_function.__name__)
#     print("slow_function.__doc__:", slow_function.__doc__)
#     print("=== Test timer ===")
#     print(slow_function())

# if __name__ == "__main__":
#     print("=== Test timer ===")
#     print(slow_function())

#     print("\n=== Test retry ===")
#     try:
#         print(flaky_api_call())
#     except Exception as e:
#         print("Final failure:", e)


if __name__ == "__main__":
    print("\n=== Test chained decorators ===")
    try:
        print("Result:", unstable_sum(10, 20))
    except Exception as e:
        print("unstable_sum failed:", e)

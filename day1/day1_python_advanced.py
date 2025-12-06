import time
import asyncio
from functools import wraps
from typing import AsyncIterator, List
from dataclasses import dataclass


# === DECORATOR: @timer ===
def timer(func):
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[timer] {func.__name__} took {end - start:.3f}s")
        return result

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"[timer] {func.__name__} took {end - start:.3f}s")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# === DATACLASS ===
@dataclass
class Item:
    name: str
    value: int


# === GENERATOR: yield từng batch ===
def batch_generator(items: List[Item], batch_size: int):
    """Yield từng batch Item."""
    batch: List[Item] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# === ASYNC GENERATOR: process items với type hints ===
@timer
async def process_items(items: List[Item]) -> AsyncIterator[str]:
    """Async generator: xử lý từng item, yield kết quả."""
    for item in items:
        await asyncio.sleep(0.1)  # giả lập I/O
        yield f"Processed {item.name} with value {item.value * 2}"


# === ASYNC FUNCTION: consume async generator ===
@timer
async def run_pipeline(items: List[Item]) -> List[str]:
    """Chạy pipeline: dùng async generator bên trong."""
    results: List[str] = []
    async for result in process_items(items):
        print(result)
        results.append(result)
    return results


# === MAIN ===
async def main():
    # Tạo data
    items = [Item(name=f"item_{i}", value=i) for i in range(1, 6)]

    # Demo generator (sync)
    print("=== Batch Generator (sync) ===")
    for batch in batch_generator(items, batch_size=2):
        print("Batch:", batch)

    # Demo async generator + decorator
    print("\n=== Async Pipeline ===")
    final_results = await run_pipeline(items)
    print("\nFinal results:", final_results)


if __name__ == "__main__":
    asyncio.run(main())

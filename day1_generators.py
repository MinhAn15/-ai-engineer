def simple_generator():
    print("Yield 1")
    yield 1
    print("Yield 2")
    yield 2
    print("Yield 3")
    yield 3


def simple_function():
    print("Return list")
    return [1, 2, 3]


def read_large_file(filename: str):
    """Đọc file dòng-by-dòng, không load hết vào RAM."""
    with open(filename, encoding="utf-8") as f:
        for line in f:
            yield line.strip()


from typing import Iterable, List


def batch_iterator(data: Iterable[int], batch_size: int) -> Iterable[List[int]]:
    """Yield từng batch có kích thước cố định."""
    batch: List[int] = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch




if __name__ == "__main__":
    # print("=== simple_function ===")
    # lst = simple_function()
    # print("Result from simple_function:", lst)

    # print("\n=== simple_generator ===")
    # gen = simple_generator()
    # print("First next:", next(gen))
    # print("Second next:", next(gen))
    # print("Third next:", next(gen))

    # print("\n=== read_large_file (demo với file nhỏ) ===")
    # for line in read_large_file(__file__):
    #     print(line)
    #     break  # demo: chỉ in 1 dòng đầu
    
    # print("\n=== generator expression ===")
    # print(squares)
    # for s in squares:
    #     print(s)

    print("\n=== batch_iterator ===")
    data = range(1, 11)  # 1..10
    for batch in batch_iterator(data, batch_size=4):
        print("Batch:", batch)
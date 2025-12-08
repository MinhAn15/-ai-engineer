import asyncio


async def fetch_data(url: str) -> str:
    """Giả lập gọi API mất 1 giây."""
    print(f"[fetch_data] Starting: {url}")
    await asyncio.sleep(1)  # giả lập chờ network
    print(f"[fetch_data] Done: {url}")
    return f"Data from {url}"


async def main():
    result = await fetch_data("https://api.example.com")
    print("Result:", result)

async def main_concurrent():
    """Gọi 3 API cùng lúc, không chờ từng cái."""
    print("\n=== Concurrent calls with gather ===")
    results = await asyncio.gather(
        fetch_data("https://api1.example.com"),
        fetch_data("https://api2.example.com"),
        fetch_data("https://api3.example.com"),
    )
    for r in results:
        print("Got:", r)


# if __name__ == "__main__":
#     import time

#     # Tuần tự (để so sánh)
#     print("=== Sequential calls ===")
#     start = time.time()
#     asyncio.run(main())
#     asyncio.run(main())
#     asyncio.run(main())
#     print(f"Sequential took: {time.time() - start:.2f}s\n")

#     # Song song
#     start = time.time()
#     asyncio.run(main_concurrent())
#     print(f"Concurrent took: {time.time() - start:.2f}s")


 

import aiohttp


async def fetch_real_url(url: str) -> str:
    """Gọi HTTP GET thật."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            return text[:100]  # chỉ lấy 100 ký tự đầu


async def main_real():
    print("\n=== Real HTTP calls ===")
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/ip",
        "https://httpbin.org/headers",
    ]
    results = await asyncio.gather(*[fetch_real_url(u) for u in urls])
    for url, data in zip(urls, results):
        print(f"{url} -> {data[:50]}...")


if __name__ == "__main__":
    asyncio.run(main_real())

import time

from contextlib import contextmanager

@contextmanager
def timer(name="Operation"):
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed_ms:.2f} ms")
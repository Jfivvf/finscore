import time


class TrackedModel:
    def __init__(self, model):
        """Оборачивает модель."""
        self.model = model
        self.calls = 0
        self.tokens = 0
        self.seconds = 0.0

    def invoke(self, messages):
        """Вызывает модель, считает токены и время."""
        start = time.perf_counter()
        response = self.model.invoke(messages)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        usage = getattr(response, "usage_metadata", None) or {}
        self.tokens += usage.get("total_tokens", 0)
        return response

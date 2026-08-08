from typing import List, Tuple, Any

class ConsolidationQueue:
    def __init__(self, max_queue_size: int = 100):
        self.queue: List[Tuple[str, str]] = []
        self.max_queue_size = max_queue_size

    def push(self, prompt_text: str, completion_text: str) -> bool:
        if len(self.queue) >= self.max_queue_size:
            return False
        self.queue.append((prompt_text, completion_text))
        return True

    def pop_batch(self, batch_size: int = 1) -> List[Tuple[str, str]]:
        batch = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        return batch


class ChatStreamProcessor:
    """
    Управляет потоком чата с авто-консолидацией выпадающих реплик.
    Оптимизирован для $O(1)$ подсчета токенов.
    """
    def __init__(self, engine: Any, max_window_tokens: int = 512, use_queue: bool = True):
        self.engine = engine
        self.max_window_tokens = max_window_tokens
        self.use_queue = use_queue
        # Формат хранения: (user_prompt, assistant_completion, token_count)
        self.chat_history: List[Tuple[str, str, int]] = []
        self.total_tokens = 0

    def process_turn(self, user_prompt: str, assistant_completion: str) -> None:
        turn_text = user_prompt + assistant_completion
        turn_tokens = len(self.engine.tokenizer.encode(turn_text, add_special_tokens=False))

        self.chat_history.append((user_prompt, assistant_completion, turn_tokens))
        self.total_tokens += turn_tokens

        # O(1) выталкивание переполненных токенов
        while self.total_tokens > self.max_window_tokens and len(self.chat_history) > 1:
            evicted_prompt, evicted_completion, evicted_tokens = self.chat_history.pop(0)
            self.total_tokens -= evicted_tokens
            
            if self.use_queue:
                self.engine.enqueue_fact(evicted_prompt, evicted_completion)
                self.engine.process_queue(batch_size=1)
            else:
                self.engine.consolidate(evicted_prompt, evicted_completion)
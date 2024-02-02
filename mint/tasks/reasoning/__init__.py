import logging
import threading
from typing import Optional, Any

from ..base import Task
from .grader import grade_answer

LOGGER = logging.getLogger("MINT")


class ReasoningTask(Task):
    task_name = "reasoning"

    def __init__(self, id: str, prompt: str, reference: str, **kwargs):
        super().__init__(**kwargs)
        self._id = id
        self._prompt = prompt.strip()
        self._reference = str(reference).strip().lower()

    def extract_answer(self, solution: str) -> Optional[str]:
        """Extract the answer from the given solution."""
        return solution.lower().strip()

    def compare_w_digits(self, reference: str, answer: str) -> bool:
        """Compare the reference and answer with digits."""
        # if reference can and answer can both be converted to floats by float()
        try:
            float(reference)
            float(answer)
            return abs(float(reference) - float(answer)) <= 0.05 * float(reference)
        except ValueError:
            return reference in answer
        except Exception:
            raise ValueError(f"Cannot compare {reference} and {answer}")

    def success(self, solution: str) -> bool:
        answer = self.extract_answer(solution)
        return self.compare_w_digits(self._reference, answer)


class FunctionThread(threading.Thread):
    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args
        self.result = None
        
    def run(self):
        self.result = self.func(*self.args)

class MATHTask(Task):
    """MATH Task for training.
    https://github.com/hendrycks/math
    """

    task_name = "reasoning"

    def __init__(self, id, question: str, answer: str, **kwargs):
        super().__init__(**kwargs)
        self._id = id
        self._prompt = question
        self._reference = answer
        self.extra = kwargs.get("extra", {})

    def success(self, solution: str) -> bool:
        # return grade_answer(solution, self._reference)
        # This is a hack to avoid hanging on some inputs
        t = FunctionThread(grade_answer, solution, self._reference)
        t.start()
        t.join(timeout=10)  # Wait for 10 seconds
        
        # Consider it as a failure if the function didn't complete within the timeout
        if t.is_alive():
            LOGGER.warning(f"Timeout on {self._id}. Solution: {solution}; Reference: {self._reference}")
            return False
        
        return t.result

    def to_dict(self) -> dict:
        """Convert the task to a dictionary."""
        return {
            "task_name": self.task_name,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "reference": self.reference,
            "extra": self.extra,
        }

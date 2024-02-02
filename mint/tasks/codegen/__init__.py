import json
import logging
from typing import List, Optional

LOGGER = logging.getLogger("MINT")

from ..base import Task
from .APPS.utils import check_correctness as apps_check_correctness
from mint.utils.exec import check_correctness

class CodeGenTask(Task):
    """Generic code generation task instance."""

    def __init__(self, id: str, prompt: str, reference: str, **kwargs):
        super().__init__(**kwargs)
        self._id = id
        self._prompt = prompt
        self._reference = reference

    def success(self, solution: str) -> bool:
        """This checks whether the given solution can complete the current task.

        Can be used to provides binary feedback.
        """
        code_to_exec = self.extract_answer(solution)
        LOGGER.debug(f"CODE_TO_EXEC:\n{code_to_exec}")
        LOGGER.debug(f"TEST_CODE:\n{self._reference}")
        res = check_correctness(
            solution_code=code_to_exec, test_code=self._reference, timeout=10
        )
        return res["success"]


class MBPPTask(CodeGenTask):
    task_name = "mbpp"

    @property
    def prompt(self) -> str:
        """Return the prompt for this task.

        MBPP prompt contains \"\"\" enclosed at both ends. Need to remove it.
        """
        return self._prompt.replace('"""', "").strip()

    def extract_answer(self, solution: str) -> Optional[str]:
        """Extract the answer from the given solution.

        Split off first block of code by scanning for class, def etc. on newlines.

        Modified from:
        https://github.com/bigcode-project/bigcode-evaluation-harness/blob/d61afde130005ecc65cf800ad8eca790a9bc2115/lm_eval/tasks/mbpp.py#L67
        """
        # STOP_WORDS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"]
        # return re.split("|".join(STOP_WORDS), solution)[0].rstrip()
        return solution


class HumanEvalTask(CodeGenTask):

    task_name = "humaneval"

    @property
    def prompt(self) -> str:
        """Return the prompt for this task.

        MBPP prompt contains \"\"\" enclosed at both ends. Need to remove it.
        """
        return "Complete the following code:\n\n" + self._prompt

    def extract_answer(self, solution: str) -> Optional[str]:
        """Extract the answer from the given solution.

        Split off first block of code by scanning for class, def etc. on newlines.

        Modified from:
        https://github.com/bigcode-project/bigcode-evaluation-harness/blob/d61afde130005ecc65cf800ad8eca790a9bc2115/lm_eval/tasks/humaneval.py#L56
        """

        # STOP_WORDS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
        # # Remove the last block of the code containing stop_words for HumanEval
        # string_list = re.split("(%s)" % "|".join(STOP_WORDS), solution)
        # # last string should be ""
        # return "".join(string_list[:-2])
        return solution


class APPSTask(CodeGenTask):

    task_name = "APPS"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # input_output dict used for APPS
        if self._reference:
            self._reference = json.loads(self._reference)
        else:
            raise ValueError("APPS task must have reference")

    def success(self, solution: str) -> bool:
        """This checks whether the given solution can complete the current task.

        Can be used to provides binary feedback.
        """
        # code_to_exec = self.extract_answer(solution)
        res = apps_check_correctness(
            in_outs=self._reference,
            generation=solution,
            timeout=10
        )
        LOGGER.debug(f"CODE_TO_EXEC:\n{solution}")
        LOGGER.debug(f"TEST_CODE:\n{self.reference}")
        LOGGER.debug(f"APPS_CHECK_CORRECTNESS: {res}")
        # res return a list of int indicating the correctness of each test case
        # True: correct, -1, -2: incorrect -> need to check if all are True
        success = all(map(lambda x: x == True, res))
        return success

import json
import sqlite3
import logging
import pandas as pd
import ast
from typing import Optional, Any, List, Mapping

from ..base import Task
from .evaluator import tsv_unescape_list, to_value_list, check_denotation

LOGGER = logging.getLogger("MINT")


def dataframe_to_sqlite_in_memory(df, table_name):
    # Connect to an in-memory SQLite database
    conn = sqlite3.connect(":memory:")
    
    # Convert DataFrame to SQLite table
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # You can return the connection if you want to use it elsewhere
    return conn

class WikiTableQuestionsTask(Task):
    task_name = "tabular"  # will re-use reasoning's ICL

    def __init__(
        self,
        id: str,
        prompt: str,
        reference: str,
        question_format: str,
        table: dict,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert question_format in ["sql", "pandas"]
        self._id = id
        self._prompt = prompt.strip()
        # Need to expand list since list is delimited by |
        self._reference = to_value_list(tsv_unescape_list(
            self._normalize(reference)
        ))
        self.table = pd.read_json(json.dumps(table))
        self.question_format = question_format

    def _normalize(self, solution: str) -> str:
        """Normalize the solution."""
        # if only , and digits are in the solution
        if solution.replace(",", "").isdigit():
            # try to remove the comma in a number
            solution = solution.replace(",", "")
        return solution

    def extract_answer(self, solution: str) -> List[str]:
        """Extract the answer from the given solution."""
        solution = self._normalize(solution)

        # attempt to get a JSON list
        try:
            solution = ast.literal_eval(solution)
            if isinstance(solution, tuple):
                solution = list(solution)
            assert isinstance(solution, list)
        except:
            solution = [solution]
        return solution

    def success(self, solution: str) -> bool:
        answer = self.extract_answer(solution)
        predicted_values = to_value_list(answer)
        LOGGER.info(f"MODEL: {predicted_values}")
        LOGGER.info(f"REFERENCE: {self._reference}")
        return check_denotation(self._reference, predicted_values)

    @property
    def reference(self) -> str:
        """Return the reference solution for the task."""
        assert hasattr(self, "_reference"), "Task does not have a reference solution."
        # The _reference might NOT be a string due to WikiTable's specific handling
        return str(self._reference)

    @property
    def user_ns(self) -> Mapping[str, Any]:
        """Return task-specific user namespace."""
        if self.question_format == "sql":
            if not hasattr(self, "conn"):
                self.conn = dataframe_to_sqlite_in_memory(self.table, "data_table")
            return {"conn": self.conn}
        elif self.question_format == "pandas":
            return {"df": self.table, "pd": pd}
        else:
            raise NotImplementedError

    def cleanup(self) -> None:
        """Cleanup the task."""
        if self.question_format == "sql" and hasattr(self, "conn"):
            self.conn.close()
            del self.conn
            LOGGER.info("SQLite connection closed.")


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

from agent_instrumented import support_agent


def test_support_agent():
    input = "Where is my order ORD-1042?"
    actual_output = support_agent(input)

    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output
    )

    evaluate(
        test_cases=[test_case],
        metrics=[
            TaskCompletionMetric(
                threshold=0.9,
                model=OllamaModel(
                    model="qwen2.5:0.5b",
                    base_url="http://localhost:11434"
                )
            )
        ]
    )
# run like this
# deepeval test run test_TaskCompletion.py
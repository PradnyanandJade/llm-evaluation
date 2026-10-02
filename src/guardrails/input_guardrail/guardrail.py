from src.llm.model import llm
from src.guardrails.schemas.guardrail_result import GuardrailResult
from src.guardrails.prompts.input_guardrail_prompt import INPUT_GUARDRAIL_PROMPT
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

class InputGuardrail:

    def __init__(self):
        self.llm = llm.with_structured_output(GuardrailResult)
        self.prompt = PromptTemplate.from_template(
            template=INPUT_GUARDRAIL_PROMPT
        )

    @traceable(name="Input Guardrail")
    def check(self,user_input : str) -> GuardrailResult:
        prompt = self.prompt.invoke({
            "user_input":user_input
            }
        )
        result = self.llm.invoke(prompt)
        return result
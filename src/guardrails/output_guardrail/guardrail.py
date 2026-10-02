from src.llm.model import llm
from src.guardrails.schemas.guardrail_result import GuardrailResult
from src.guardrails.prompts.output_guardrail_prompt import OUTPUT_GUARDRAIL_PROMPT
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

class OutputGuardrail:

    def __init__(self):
        self.llm = llm.with_structured_output(GuardrailResult)
        self.prompt = PromptTemplate.from_template(
            template=OUTPUT_GUARDRAIL_PROMPT
        )

    @traceable(name="Output Guardrail")
    def check(self,output : str) -> GuardrailResult:
        prompt = self.prompt.invoke({
            "output":output
            }
        )
        result = self.llm.invoke(prompt)
        return result
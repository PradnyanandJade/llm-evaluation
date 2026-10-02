from src.llm.model import llm
from src.prompts.generator_prompt import GENERATOR_PROMPT
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

GENERATOR_PROMPT_TEMPLATE = PromptTemplate.from_template(
    template=GENERATOR_PROMPT
)

class Generator:

    def __init__(self):
        pass

    @traceable(name="Generator")
    def generate(self,question:str,context:str):
        prompt = GENERATOR_PROMPT_TEMPLATE.invoke({
            "question":question,
            "context":context
        })
        return llm.invoke(prompt).content

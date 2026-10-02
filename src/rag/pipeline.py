from src.rag.retrieval.reranker import Reranker
from src.rag.generation.generator import Generator
from src.guardrails.input_guardrail.guardrail import InputGuardrail
from src.guardrails.output_guardrail.guardrail import OutputGuardrail
from langsmith import traceable


class RAGPipeline:

    def __init__(self):
        self.retriever = Reranker()
        self.generator = Generator()
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()

    @traceable(name="RAG Pipeline")
    def run(self,question:str):
        input_result = self.input_guardrail.check(
            user_input=question
        )
        if input_result.decision == "block":
            return {
                "answer": "I can't help with that request.",
                "documents": [],
                "blocked": True,
                "block_reason": input_result.reason
            }
        documents = self.retriever.retrieve(query=question)
        context = "\n\n".join(
            document.page_content
            for document in documents
        )
        answer = self.generator.generate(
            question=question,
            context=context
        )
        
        output_result = self.output_guardrail.check(
            output=answer
        )
        if output_result.decision == "block":
            return {
                "answer": "I can't provide that response.",
                "documents": documents,
                "blocked": True,
                "block_reason": output_result.reason
            }
        
        return {
            "answer":answer,
            "documents":documents,
            "blocked":False
        }
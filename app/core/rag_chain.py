

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based on the provided context.

Do not make up information. Only use the provided context.

Context:
{context}

Question: {question}

Answer:
"""

class RAGChain:

    def __init__(self, vector_store_service=None, ragas_evaluator=None):
        from langchain_core.documents import Document
        from langchain_ollama import OllamaLLM
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from app.core.ragas_evaluator import RAGASEvaluator
        from app.core.vector_store import VectorStoreService

        settings = get_settings()

        self.vector_store_service = vector_store_service or VectorStoreService()
        self.retriever = self.vector_store_service.get_retriever()

        self.evaluator = ragas_evaluator or RAGASEvaluator()

        self.llm = OllamaLLM(
            model=settings.llm_model,
            temperature=settings.temperature
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        def format_docs(docs: list[Document]) -> str:
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        try:
            answer = self.chain.invoke(question)
            logger.info("Question answered successfully")
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def query_with_sources(self, question: str) -> dict:
            try:
                answer = self.chain.invoke(question)
                source_docs = self.retriever.invoke(question)

                sources = [
                    {
                        "content": (doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content),
                        "metadata": doc.metadata
                    } 
                    for doc in source_docs
                ]
                logger.info("Question answered successfully")
                return {
                    "answer": answer,
                    "sources": sources
                }
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise

    async def aquery(self, question: str) -> str:
        try:
            answer = await self.chain.ainvoke(question)
            logger.info("Question answered successfully")
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    async def aquery_with_sources(self, question: str) -> dict:
        try:
            answer = await self.chain.ainvoke(question)
            source_docs = await self.retriever.ainvoke(question)

            sources = [
                {
                    "content": (doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content),
                    "metadata": doc.metadata
                } 
                for doc in source_docs
            ]
            logger.info("Question answered successfully")
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    async def aquery_with_evaluation(self, question: str) -> dict:
        try:
            result = await self.aquery_with_sources(question=question)
            answer = result["answer"]
            sources = result["sources"]

            contexts = [source["content"] for source in sources]

            try:
                evaluation = await self.evaluator.aevaluate(
                    question=question, answer=answer, contexts=contexts
                )
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness = {evaluation.get('faithfulness', 'N/A')}"
                    f"answer_relevancy = {evaluation.get('answer_relevancy', 'N/A')}"
                )
            except Exception as e:
                logger.warning(
                    f"Evaluation failed - {e}"
                )

                evaluation = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "evaluation_time_ms": None,
                    "error": str(e)
                }

            return {"answer": answer, "sources": sources, "evaluation": evaluation}
        except Exception as e:
            logger.error(f"Evaluation failed - {e}")
            raise
        

    def stream(self, question:str):
        try:
            for chunk in self.chain.stream(question):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            raise
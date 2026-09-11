import time
import asyncio
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:

    def __init__(self):
        from langchain_ollama import OllamaLLM, OllamaEmbeddings

        logger.info("Initialising RAGAS Evaluator")

        self.settings = get_settings()

        eval_llm_model = self.settings.ragas_llm_model or self.settings.llm_model
        eval_llm_temperature = self.settings.ragas_llm_temperature or self.settings.temperature
        eval_embedding_model = self.settings.eval_embedding_model or self.settings.embedding_model

        self.llm = OllamaLLM(
            model=eval_llm_model,
            temperature=eval_llm_temperature
        )

        self.embeddings = OllamaEmbeddings(
            model=eval_embedding_model
        )

        from ragas.metrics import faithfulness, answer_relevancy

        self.metrics = [
            faithfulness,
            answer_relevancy
        ]

    async def aevaluate(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:

        logger.debug(f"Starting evaluation or question: {question[:100]}...")
        start_time = time.time()

        try:
            dataset = self._prepare_dataset(question, answer, contexts)

            result = await asyncio.to_thread(
                self._evaluate_with_timeout,
                dataset
            )

            evaluation_time_ms = (time.time() - start_time) * 1000

            scores = {
                "faithfulness": float(result["faithfulness"]) if "faithfulness" in result else None,
                "answer_relevancy": float(result["answer_relevancy"]) if "answer_relevancy" in result else None,
                "evaluation_time_ms": round(evaluation_time_ms, 2),
                "error": None
            }

            if self.settings.ragas_log_result:
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness = {scores.get('faithfulness', 'N/A')}"
                    f"answer_relevancy = {scores.get('answer_relevancy', 'N/A')}"
                )

            return scores

        except Exception as e:
            logger.warning(f"Evaluation failed - {e}", exc_info=True)
            return self._handle_evaluation_error(e)

    def _prepare_dataset(self, question: str, answer: str, contexts: list[str]):

        from datasets import Dataset

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }

        return Dataset.from_dict(data)

    def _evaluate_with_timeout(self, dataset) -> dict[str, Any]:

        from ragas import evaluate

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )

        return result.to_pandas().to_dict("records")[0]

    def _handle_evaluation_error(self, e):
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "evaluation_time_ms": None,
            "error": str(e)
        }
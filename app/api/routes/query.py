import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import EvaluationScores, QueryRequest, QueryResponse, SourceDocument
from app.core.rag_chain import RAGChain
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question"
)
async def query(request: QueryRequest) -> QueryResponse: 
    start_time = time.time()

    try:
        rag_chain = RAGChain()

        if request.enable_evaluation:
            result = await rag_chain.aquery_with_evaluation(question=request.question)

            sources = (
                [
                    SourceDocument(content=source["content"], metadata=source["metadata"])
                    for source in result["sources"]
                ]
            )

            answer = result["answer"]
            evaluation = EvaluationScores(**result["evaluation"])

        elif request.include_sources:
            result = await rag_chain.aquery_with_sources(question=request.question)
            sources = (
                [
                    SourceDocument(content=source["content"], metadata=source["metadata"])
                    for source in result["sources"]
                ]
            )
            answer = result["answer"]
            evaluation = None
        else:
            answer = await rag_chain.aquery(question=request.question)
            sources = None
            evaluation = None

        processing_time_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time_ms=round(processing_time_ms, 2),
            evaluation=evaluation
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@router.post(
    "/stream",
    summary="Ask a question (streaming)"
)
async def query_stream(request: QueryRequest) -> StreamingResponse:

    try:
        rag_chain = RAGChain()

        async def generate():
            try:
                for chunk in rag_chain.stream(question=request.question):
                    yield chunk
            except Exception as e:
                yield f"\n\nError: {str(e)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@router.post(
    "/search",
    summary="Search documents"
)
async def search_documents(query: QueryRequest) -> dict:
    try:
        vector_store = VectorStoreService()
        results = vector_store.search_with_score(query=query.question)

        documents = [
            {
                "content":doc.page_content,
                "metadata":doc.metadata,
                "relevance_score": score
            }
            for doc, score in results
        ]

        return {
            "query": query.question,
            "results": documents,
            "count": len(documents)
        }
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )

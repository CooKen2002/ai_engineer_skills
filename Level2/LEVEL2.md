### Tầng 2 — Kỹ năng triển khai thực tế (điều phân biệt fresher vs. có thể làm việc được)

Đây là phần **xuất hiện lặp lại nhiều nhất** trong các JD bạn gửi — gần như là "signature" của một AI Engineer thực chiến năm 2025-2026:

**RAG pipeline production:**

- Vector database: Pinecone, Weaviate, Milvus, Chroma, Qdrant, Elasticsearch
- Reranking, đánh giá RAG (faithfulness, relevance, context precision/recall)

**Agent frameworks:**

- LangChain, LlamaIndex, LangGraph, AutoGen
- Tool/function calling, multi-agent workflow (planner → executor → evaluator)
- **MCP (Model Context Protocol)** — đây là kỹ năng mới nổi, xuất hiện ở rất nhiều JD gần đây: viết MCP server, expose tool/API cho LLM

**Backend để expose model:**

- FastAPI (ưu tiên hơn Flask trong hầu hết JD), REST API design
- Docker cơ bản, hiểu container hóa

**Serving & optimization:**

- vLLM, Ollama, HuggingFace TGI, LM Studio — self-host LLM
- Quantization, batching cơ bản để tối ưu latency/cost
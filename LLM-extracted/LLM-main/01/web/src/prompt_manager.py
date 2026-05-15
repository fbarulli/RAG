from src.logger_config import logger

def build_prompt(question: str, records: list) -> str:
    context_template: str = "Q: {question}\nA: {text}"
    prompt_template: str = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

    try:
        context_entries: list[str] = [
            context_template.format(
                question=hit['_source']['question'], 
                text=hit['_source']['text']
            ) 
            for hit in records
        ]
        
        context: str = "\n\n".join(context_entries)
        prompt: str = prompt_template.format(question=question, context=context)
        return prompt
        
    except Exception as e:
        logger.error(f"Failed to build prompt template: {e}")
        raise

import os
import chromadb
import json
from openai import AsyncOpenAI
import re
from chromadb.utils import embedding_functions
import chat.assistant as assistant
from db_operations import store_conversation_in_db, get_conversation_history_from_db
from codebase.open_ai_token_counter import open_ai_truncator
from chat.generate_html import generate_html

# Load environment variables
api_key = os.getenv('OPENAI_API_KEY')
absolutepath = os.getenv("ABSOLUTE_PATH")
open_ai_client = AsyncOpenAI(api_key=api_key)

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

# Summarize early messages
async def summarize_early_exchanges(messages: list[dict], num_exchanges: int = 4) -> str:
    if len(messages) <= num_exchanges:
        return ""

    early_messages = messages[:num_exchanges * 2]
    summary_prompt = "Summarize the following conversation concisely:\n\n"
    for msg in early_messages:
        summary_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

    truncated_summary_prompt = await open_ai_truncator(
        text=summary_prompt,
        max_tokens=20000,
        model="gpt-4.1-mini"
    )
    summary_response = await open_ai_client.responses.create(
        model="gpt-4.1-mini",
        max_output_tokens=2500,
        input=f"You are an AI assistant tasked with summarizing conversations. Provide a brief summary. Here is the prompt: \n{truncated_summary_prompt}\n"
    )
    return summary_response.output[0].content[0].text

def query_chroma_db(collection, query):
    results = collection.query(query_texts=query, n_results=8)
    if not results['documents'][0]:
        return "No relevant codebase content found for the generated query."
    else:
        return "\n".join([f"results: {doc}" for doc in results['documents'][0]])

# Main Chat Function
async def chat(email_id: str, project_id: str, summary: str, user_question: str) -> str:
    try:
        client = chromadb.PersistentClient(path=absolutepath)
        collection = client.get_collection(name=project_id, embedding_function=openai_ef)
        conversation_messages = await get_conversation_history_from_db(email_id, project_id)
        summary = await open_ai_truncator(text=summary, max_tokens=100000, model="gpt-4.1-mini")

        system_message = [
            {"role": "system", "content": "You are a senior software architect and expert in scalable backend systems, cloud-native applications, and vector database technology. Your task is to analyze codebases and technical systems in exhaustive detail. You excel at systematically breaking down complex systems into their individual components and explaining the purpose, functionality, dependencies, and practical considerations of each component in a clear, structured, and comprehensive manner."},
            {"role": "system", "content": f"Here is the summary of a complex codebase: {summary}"}
        ]

        api_messages = []
        if conversation_messages:
            early_summary = await summarize_early_exchanges(conversation_messages[:-2])
            if early_summary:
                system_message.append({"role": "system", "content": f"Earlier conversation summary: {early_summary}"})
            for i, msg in enumerate(conversation_messages[-2:]):
                role = "user" if i % 2 == 0 else "assistant"
                api_messages.append({"role": role, "content": msg["content"]})
        api_messages.append({"role": "user", "content": user_question})

        # Step 1: Generate a query to run on ChromaDB
        query_prompt = system_message + api_messages + [
            {"role": "system", "content": "Based on the above context, generate a search query to find relevant code snippets or documents. Only generate a checklist-related query if the user asks for a checklist."},
            {"role": "system", "content": "If the user asks to create a checklist, generate a checklist-related query. Make sure the query contains either of the keyword - 'checklist', 'create a checklist' or 'make a checklist'."}
        ]

        query_response = await open_ai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=query_prompt,
            max_tokens=1500,
            temperature=0.3
        )
        rag_query = query_response.choices[0].message.content

        # Step 2: Retrieve from DB
        db_result = query_chroma_db(collection, rag_query)
        retrieved = await open_ai_truncator(text=db_result, max_tokens=20000, model="gpt-4.1-mini")

        # Check if checklist is requested
        checklist_tool_call = None
        if any(kw in rag_query.lower() for kw in ["checklist", "create a checklist", "make a checklist"]):
            checklist_tool_call = {
                "tool": "CHECKLIST_ASSISTANT",
                "tool_input": rag_query
            }

        # Step 3: Compose detailed final answer
        final_prompt = system_message + api_messages + [
            {"role": "system", "content": f"Here is the search query used to find relevant codebase information: {rag_query}"},
            {"role": "system", "content": f"The following information was retrieved from the codebase using that query: {retrieved}"},
            {"role": "system", "content": f"Here is the original question that was asked by the user: {user_question}"},
            {"role": "user", "content": """
                Please analyze all the above context and provide a comprehensive, technically precise, and implementation-ready answer to the original question.
                If the answer involves explaining a collection of items (such as components, files, functions, modules, database tables, services, or configuration parameters), ensure you:
                - Enumerate each item explicitly as a clear list or structured format
                - Provide a detailed explanation of the purpose, functionality, dependencies, and relevant implementation details of each individual item
                - Avoid summarizing multiple items together or skipping components, even if they seem minor
                - Where applicable, include examples, configuration details, relationships to other components, and real-world usage considerations

                Your goal is to make the answer exhaustive, systematic, and useful for a developer or architect seeking deep understanding of every component.
            """},
            {"role": "user", "content": "Where possible, present your output in a structured, itemized format (such as a numbered list or table) to improve clarity and navigability."}
        ]

        final_response = await open_ai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=final_prompt,
            max_tokens=5000,
            tool_choice="required" if checklist_tool_call else "none",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "CHECKLIST_ASSISTANT",
                        "description": "IMPORTANT USAGE INSTRUCTIONS: 1) Only use when a user explicitly asks to CREATE A CHECKLIST. 2) This tool formats the DETAILED content YOU provide. 3) YOU MUST include SPECIFIC examples, version numbers, concrete implementation steps, and technical details in your checklist content. 4) AVOID generic items like 'Update database' - instead use specifics like 'Upgrade PostgreSQL from 13.4 to 15.2, implementing the new JSON path operators and performance improvements'. 5) Include actual technologies, versions, commands, and implementation steps where applicable. 6) Do Not call this tool if the user did not ask for it explicitly",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "Title": {
                                    "type": "string",
                                    "description": "A suitable title for the checklist."
                                },
                                "checklist": {
                                    "type": "string",
                                    "description": "Your COMPLETE, HIGHLY DETAILED checklist with SPECIFIC version numbers, technology names, commands, implementation steps, and technical details. Generic items will NOT be accepted. Example: Instead of 'Update web framework' use 'Upgrade Django from 3.2 to 4.2, implementing async views and the new template engine features'."
                                }
                            },
                            "required": ["Title", "checklist"]
                        }
                    }
                }
            ]
        )

        if (final_response.choices[0].finish_reason == "tool_calls"):
            ai_response = json.loads(final_response.choices[0].message.tool_calls[0].function.arguments).get("checklist", "")
            await assistant.assistant_function(
                            project_id=project_id,
                            assistant_action="CHECKLIST_ASSISTANT",
                            content=ai_response
                        )
        else:
            ai_response = final_response.choices[0].message.content

        await store_conversation_in_db(email_id, project_id, "user", user_question)
        await store_conversation_in_db(email_id, project_id, "assistant", ai_response)

        ai_response = await generate_html(ai_response)
        return ai_response

    except Exception as e:
        return str(e)

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json
from anthropic import Anthropic
from db_operations import  store_conversation_in_db, get_conversation_history_from_db 
import chromadb.utils.embedding_functions as embedding_functions
import chat.assistant as assistant
from codebase.anthropic_token_counter import anthropic_truncator
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
absolutepath = os.getenv("ABSOLUTE_PATH")



# Initialize Anthropic client
anthropic = Anthropic(api_key=anthropic_api_key)

async def summarize_early_exchanges(messages: list[dict], num_exchanges: int = 4) -> str:
    if len(messages) <= num_exchanges :  # If we don't have enough messages to summarize
        return ""
    
    early_messages = messages[:num_exchanges * 2]  # Get the first 4 exchanges (8 messages)
    summary_prompt = "Summarize the following conversation concisely:\n\n"
    for msg in early_messages:
        summary_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
    
    truncated_summary_prompt = anthropic_truncator(text =summary_prompt )
    summary_response = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        system=[{"type": "text", "text": "You are an AI assistant tasked with summarizing conversations. Provide a brief summary."}],
        messages=[{"role": "user", "content": truncated_summary_prompt}]
    )
    
    return summary_response.content[0].text

async def synthesize_information(retrieved_info, user_query):
    synthesis_response = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=7500,
        system=[{"type": "text", "text": "Analyze the following information retrieved from multiple files. Synthesize the information needed to answer a user query."}],
        messages=[{"role": "user", "content":  f'user_query {user_query}  retrived data  {retrieved_info}' },
                  ]
    )
    return synthesis_response.content[0].text

async def chat(email_id: str, project_id: str, summary: str, user_question: str) -> str:
    try:

        summary = anthropic_truncator(text =  summary, max_tokens=130000, model="claude-3-7-sonnet-20250219")

        client = chromadb.PersistentClient(path= absolutepath)

        

        # Get the collection for the project
        collection_name = project_id
        

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key = os.getenv('OPENAI_API_KEY'),
                    model_name="text-embedding-3-large")
        collection = client.get_collection(name=collection_name, embedding_function=openai_ef)

        

        def query_chroma_db(query):
            results = collection.query(
                query_texts=query,
                n_results=4
            )
            
            context = ""
            for result in results['documents'][0]:
                context += f"results: {result}\n"
              
              
            
            return context  

        # Get the relevant conversation history from the database
        conversation_messages = await get_conversation_history_from_db(email_id, project_id)
        
        

        # Prepare the messages for the API call
        api_messages = []

        # Prepare the system message
        system_message = [
            {"type": "text", "text": f""" You are an expert software Developer who can analyzing codebases.
                                        1)Go through the provided summary
                                        2) Then Study the User question and come-up with a list of files that may have details that can address the user question.
                                        3)You have been provided with a query tool. that you can call a maximum of 2 times. 
                                        4) Based on that plan, formulate a  Query that will Identify key terms, concepts, or entities that are central to the query..  This query should be designed to retrieve code snippets that are most likely to answer the user's question.
                                        5) Call the query chroma Tool if needed.
                                        6) Remember the code can be in multiple files and places, you need to determine when to query for additional context
                                        7) Do not call the checklist assistant tool without the user explicitly asking for it. If the user asks for checklist. call the checklist assistant.
                                        8) Do not recurisvely call Querycodebase tool more than 2 times, if you need to do that end the turn then ask the user to look up more files of searches"""},
            {"type": "text", "text": f"Here is the summary of a complex codebase: {summary}", "cache_control": {"type": "ephemeral"}}
            
        ]
        
        # Add initial message with summary if conversation history is empty
        if not conversation_messages:
            api_messages.append({"role": "user", "content": user_question})
            # await store_conversation_in_db(email_id, project_id, "user", user_question)
        else:
            # Summarize the early part of the conversation
            early_summary = await summarize_early_exchanges(conversation_messages[:-2])  # Summarize all but last exchange
            
            if early_summary:
                system_message.append({"type": "text", "text": f"Earlier conversation summary: {early_summary}"})
            
            # Ensure the conversation history alternates correctly
            for i, msg in enumerate(conversation_messages[-2:]):  # Add only the last exchange
                if i % 2 == 0:
                    api_messages.append({"role": "user", "content": msg["content"]})
                else:
                    api_messages.append({"role": "assistant", "content": msg["content"]})
            
            # Add the new user question
            api_messages.append({"role": "user", "content": user_question})
            #await store_conversation_in_db(email_id, project_id, "user", user_question)

        # Print api_messages for debugging
        print("API Messages:", api_messages)

        
        
        # Define the Chroma DB tool
        chroma_tool = [
                         {
                             "name" : "Querycodebase",
                             "description"  : "Execute a query against a  vector database to retrieve specific information at a code level. This returns the actual code in the project. It takes the Query and does a Vector Search",
                             "input_schema"  : {
                                                  "type" : "object", 
                                                   "properties": { 
                                                    "Query" : {
                                                                        "type" : "string",
                                                                        "description" : "The query to search for in the database, the Query must be specific in nature with references to the  specific files to search for."
                                                                      }
                                                                 },

                                 "required" : ["Query"]
                                     
                                 }
                            }
                        
                         ] 
        # Define the Assistant tool
        assistant_tool = [
                            {
                                "name": "CHECKLIST_ASSISTANT",
                                "description": "IMPORTANT USAGE INSTRUCTIONS: 1) Only use when a user explicitly asks to CREATE A CHECKLIST. 2) This tool formats the DETAILED content YOU provide. 3) YOU MUST include SPECIFIC examples, version numbers, concrete implementation steps, and technical details in your checklist content. 4) AVOID generic items like 'Update database' - instead use specifics like 'Upgrade PostgreSQL from 13.4 to 15.2, implementing the new JSON path operators and performance improvements'. 5) Include actual technologies, versions, commands, and implementation steps where applicable. 6) Do Not call this tool if the user did not ask for it explicitly",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {


                                        "Title": {
                                            "type": "string",
                                            "description": " A Suitable title for this checklist"
                                        },

                                        "checklist": {
                                            "type": "string",
                                            "description": "Your COMPLETE, HIGHLY DETAILED checklist with SPECIFIC version numbers, technology names, commands, implementation steps, and technical details. Generic items will NOT be accepted. Example: Instead of 'Update web framework' use 'Upgrade Django from 3.2 to 4.2, implementing async views and the new template engine features'."
                                        }
                                    },
                                    "required": ["checklist"]
                                }
                            }
                            
                         ] 

        # Generate a response using Anthropic with tools

        # This loop will keep running until we get an answer (not a tool_call)
        # Generate a response using Anthropic with tools
        tools = chroma_tool + assistant_tool
        while True: 
            # Always include system messages at each turn shifted to haiku
            response = anthropic.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=2000,
                temperature= 0.01,
                system=system_message,
                messages=api_messages,
                tools=tools,
                tool_choice={"type": "auto"}
            )
            print("API Return:", response)
            
            # If there's no tool use, break the loop
            if not any(content.type == "tool_use" for content in response.content):
                ai_response = response.content[0].text
                break

            # Add the assistant's response to the message history
            assistant_response = {"role": "assistant", "content": []}
            
            for content in response.content:
                if content.type == "text":
                    assistant_response["content"].append({"type": "text", "text": content.text})
                elif content.type == "tool_use":
                    assistant_response["content"].append({
                        "type": "tool_use",
                        "name": content.name,
                        "input": content.input,
                        "id": content.id
                    })
            
            api_messages.append(assistant_response)
            
            # Process each tool call
            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    
                    if tool_name == "Querycodebase":
                        query = tool_input["Query"]
                        tool_result = query_chroma_db(query)
                        truncated_tool_result = anthropic_truncator(text=tool_result, max_tokens=20000)
                        
                        # Add the tool result as a user message with formatted content
                        api_messages.append({
                            "role": "user", 
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": truncated_tool_result
                                }
                            ]
                        })
                        
                    elif tool_name == "CHECKLIST_ASSISTANT":
                        result = await assistant.assistant_function(
                            project_id=project_id,
                            assistant_action=tool_name,
                            content=tool_input
                        )
                        
                        # Add the tool result as a user message
                        api_messages.append({
                            "role": "user", 
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": str(result)
                                }
                            ]
                        })
                        
                    else:
                        # Handle unknown tools
                        api_messages.append({
                            "role": "user", 
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": "Unknown or unhandled tool."
                                }
                            ]
                        })

        # Now ai_response has the user-facing answer:
        await store_conversation_in_db(email_id, project_id, "user", user_question)
        await store_conversation_in_db(email_id, project_id, "assistant", ai_response)
        return ai_response

    except Exception as e:
        return str(e)

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
            {"type": "text", "text": f""" You are an expert software Developer and Vector database expert tasked with analyzing codebases.
                                        1) Go through the provided summary
                                        2) Then Study the User question and come-up with a list of files that may have details that can address the user question
                                        3) Based on that, formulate a  Query that will Identify key terms, concepts, or entities that are central to the query..  This query should be designed to retrieve code snippets that are most likely to answer the user's question.
                                        4) Call the query chroma Tool with the query """},
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

        response = anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",  # Update to the latest available model
            max_tokens=2000,
            system=system_message,
            messages=api_messages,
            tool_choice = {"type": "tool", "name": "Querycodebase"},
            tools=chroma_tool
        )

        print(response)
        ai_response =[]
          
        try:
            if response.stop_reason == 'tool_use':
               rag_query = response.content[0].input
               response_type = "tool_call"
            else:
             ai_response = response.content[0].text 
             response_type = "ai_response"

        except Exception as e:
            return str(e)

        if response_type == "tool_call"  :
            query = rag_query["Query"]
            db_result = query_chroma_db(query)
            retrieved_db_result  = anthropic_truncator(text =db_result, max_tokens = 20000 )
            full_response = f"\n\nRetrieved information:\n```\n{db_result}\n```\n\n"

            # Feed the retrieved information back into the model for a final response
            final_system_message = [
                {"type": "text", "text": "You are an AI assistant tasked with gerating an answer to a user question on codebases. You CANNOT query the codebase directly. A previous Agent has looked at the data and come up with a query and the results of that query are provided to you. Use only the retrieved information to provide a comprehensive response. If you can't answer with high confidence, ask the user for more details. The only external tool available to you is a checklist creator, which should only be used when explicitly requested. Use the retrieved information to provide a comprehensive response.  If you cant answer the question in high confidence ask the user to provide more details on where the code could be located. IMPORTANT: Your Response should be a neatly formated Markdown."},
                {"type": "text", "text": f"Here is the query generated by the previous agent: {query}"},
                {"type": "text", "text": f"Here is the summary of the codebase: {summary}", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": f"Retrieved information from the code base :\n{retrieved_db_result}"},
                {"type": "text", "text": f"DO NOT CALL without the user explicitly asking, ask the user always before a tool call. NOTE: you must provide a detailed description when calling the tool. Tools can only extract a schema from the content you provide"}
            ]

            final_response = anthropic.messages.create(
                model="claude-3-5-sonnet-20240620",  # Update to the latest available model
                max_tokens=6000,
                system=final_system_message,
                messages=api_messages,
                tool_choice = {"type": "auto"},
                tools = assistant_tool
            )

            ai_response =[]
          
           
            if final_response.stop_reason == 'tool_use':
                assistant_query = final_response.content[1].input
                assistant_action = final_response.content[1].name
                response_type = "tool_call"
                response = await  assistant.assistant_function(project_id= project_id,   assistant_action = assistant_action, content = assistant_query  )
                ai_response =  final_response.content[0].text 

            else:
                ai_response =  final_response.content[0].text 
                response_type = "ai_response"

            #ai_response = final_response.content[0].text 

        else:
            pass 

        # Print the final response
        print(ai_response)
        last_response = ai_response

        # Add the AI's response to the conversation history
        await store_conversation_in_db(email_id, project_id, "user", user_question)
        await store_conversation_in_db(email_id, project_id, "assistant", last_response)
        
        return last_response
    except Exception as e:
        return str(e)


import os
from openai import AsyncOpenAI

api_key = os.getenv('OPENAI_API_KEY')
open_ai_client = AsyncOpenAI(api_key=api_key)

async def generate_html(ai_response):
    # Generate HTML-formatted response using GPT-4.1-mini
    html_prompt = f"""
    You are an expert AI assistant that generates perfectly valid and complete HTML pages.

    TASK:
    - Take the following response content and generate a full HTML document that includes:
        - DOCTYPE declaration.
        - <html>, <head>, and <body> sections.
        - A <title> element.
        - Properly structured and formatted content, wrapping text in <p>, <h1>, <h2>, <ul>, <li>, <pre>, <code>, etc. where appropriate.
        - Escape any characters that need escaping in HTML.

    IMPORTANT:
    - Double-check your output: After generating the HTML, internally test/verify it to ensure that if someone saves this output to an HTML file and opens it in a modern browser, it renders without any syntax errors.
    - If your generated output has any issues, regenerate it **inside the model’s own response** until it’s correct. Only return the final, error-free HTML.

    CONTENT TO FORMAT:
    \"\"\"
    {ai_response}
    \"\"\"
    """

    html_response = await open_ai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant that generates perfectly valid, complete, and error-free HTML documents from raw text responses."},
            {"role": "user", "content": html_prompt}
        ],
        max_tokens=5000,
        temperature=0.2
    )

    html_output = html_response.choices[0].message.content
    return html_output
# Flask web interface for real-time Azure Search Index testing
# Save this code to a file (e.g., app.py) and run it from the terminal: python app.py
from flask import Flask, render_template_string, request
import os
import dotenv

dotenv.load_dotenv()

app = Flask(__name__)

# Set your Azure Search Index credentials here or use environment variables
AZURE_SEARCH_KEY = os.getenv('SEARCH_ADMIN_KEY')
AZURE_SEARCH_ENDPOINT = os.getenv('SEARCH_SERVICE_ENDPOINT')
AZURE_SEARCH_INDEX = os.getenv('AZURE_SEARCH_INDEX', 'insurance-documents-index')

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Azure Search Index Tester</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
        .container { max-width: 600px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #e0e0e0; padding: 32px; }
        h2 { color: #2d3e50; }
        label { font-weight: bold; }
        input[type=text], textarea { width: 100%; padding: 8px; margin: 8px 0 16px 0; border: 1px solid #ccc; border-radius: 4px; }
        button { background: #0078d4; color: #fff; border: none; padding: 10px 24px; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background: #005fa3; }
        .result { background: #f0f6ff; border-left: 4px solid #0078d4; padding: 16px; margin-top: 24px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Azure Search Index Tester</h2>
        <form method="post">
            <label for="query">Enter your search query:</label>
            <input type="text" id="query" name="query" required value="{{ query|default('') }}">
            <button type="submit">Search</button>
        </form>
        {% if result %}
        <div class="result">
            <strong>Results:</strong>
            <div>{{ result|safe }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''


# Azure Cognitive Search integration
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

def search_azure_index(query):
    if not (AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY and AZURE_SEARCH_INDEX):
        return "Azure Search configuration is missing. Please set SEARCH_SERVICE_ENDPOINT, SEARCH_ADMIN_KEY, and AZURE_SEARCH_INDEX."
    try:
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        results = search_client.search(query)
        output = []
        for result in results:
            doc = dict(result)
            # Professional HTML card for each result
            content = doc.get('content', '')
            if not isinstance(content, str):
                content = str(content) if content is not None else ''
            show_ellipsis = len(content) > 1200
            card = f"""
            <div style='border:1px solid #e0e0e0; border-radius:8px; margin-bottom:24px; padding:16px; background:#fafcff;'>
                <h3 style='margin-top:0;color:#0078d4;'>{doc.get('title', doc.get('file_name', 'Document'))}</h3>
                <p><strong>File Name:</strong> {doc.get('file_name','')}</p>
                <p><strong>Category:</strong> {doc.get('category','')}</p>
                <p><strong>Chunk:</strong> {doc.get('chunk_id', '')} / {doc.get('chunk_count', '')}</p>
                <p><strong>Score:</strong> {round(doc.get('@search.score', 0), 2)}</p>
                <p><strong>Content:</strong></p>
                <div style='background:#f4f6fa; border-radius:4px; padding:12px; font-family:monospace; white-space:pre-wrap; max-height:300px; overflow:auto;'>
                    {content[:1200]}{('...' if show_ellipsis else '')}
                </div>
            </div>
            """
            output.append(card)
        if not output:
            return "<div style='color:#888;'>No results found.</div>"
        return "\n".join(output)
    except Exception as e:
        return f"Error querying Azure Search: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    query = ''
    if request.method == 'POST':
        query = request.form['query']
        result = search_azure_index(query)
    return render_template_string(HTML_TEMPLATE, result=result, query=query)

# To run the app, save this code to a file and run it from the terminal:
if __name__ == '__main__':
    app.run(debug=True, port=5000)
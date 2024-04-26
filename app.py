from flask import Flask, render_template, request # type: ignore
import re
import chromadb # type: ignore
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initializing chromaDB
client = chromadb.PersistentClient(path="searchengine_database") #_test_db
collection = client.get_collection(name="search_engine") #test_collection
# collection_name = client.get_collection(name="search_engine_FileName")
model_name = "paraphrase-MiniLM-L3-V2"
model = SentenceTransformer(model_name, device="cpu")

def extract_id(id_list):
    new_id_list = []
    for item in id_list:
        match = re.match(r'^(\d+)', item)
        if match:
            extracted_number = match.group(1)
            new_id_list.append(extracted_number)
    return new_id_list

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_query = request.form['search_query']
        search_query = clean_data(search_query)
        query_embed = model.encode(search_query).tolist()

        search_results = collection.query(query_embeddings=query_embed, n_results=10,include=['documents'])
        id_list = search_results['ids'][0]

        id_list = extract_id(id_list)

        documents=search_results.get('documents',[])

        return render_template('results.html', id_list=id_list,documents=documents)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

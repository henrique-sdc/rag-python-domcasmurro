import re
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk

# Caso não tenha os pacotes do nltk baixados, descomente a linha abaixo para baixar o tokenizer:
#nltk.download('punkt')


def load_text(file_path):
    with open(file_path, 'r', encoding='latin1') as f:
        text = f.read()
    return text


def split_into_chapters(text):
    pattern = re.compile(r'(Cap[ií]tulo\s+\d+)', re.IGNORECASE)
    splits = pattern.split(text)
    
    if len(splits) > 1:
        chapters = []
        if not pattern.search(splits[0]):
            intro = splits[0].strip()
            if intro:
                chapters.append(("Introdução", intro))
            splits = splits[1:]
        
        for i in range(0, len(splits), 2):
            title = splits[i].strip()
            content = splits[i+1].strip() if (i+1) < len(splits) else ""
            chapters.append((title, content))
        return chapters
    else:
        words = text.split()
        blocks = []
        block_size = 500
        for i in range(0, len(words), block_size):
            block = " ".join(words[i:i+block_size])
            blocks.append((f"Bloco {i//block_size + 1}", block))
        return blocks


def create_embeddings(chunks, model):
    names = []
    texts = []
    for name, content in chunks:
        names.append(name)
        texts.append(content)
    
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings)
    return names, embeddings, texts


def build_index(embeddings, ef=50, M=16):

    dim = embeddings.shape[1]
    num_elements = embeddings.shape[0]
    
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=200, M=M)
    index.add_items(embeddings, np.arange(num_elements))
    index.set_ef(ef)
    
    return index


def answer_question(question, model_embedding, index, names, texts, qa_pipelines):

    q_embedding = model_embedding.encode([question])
    
    labels, distances = index.knn_query(q_embedding, k=1)
    idx = labels[0][0]
    context = texts[idx]
    block_name = names[idx]
    
    results = {}
    for model_name, qa in qa_pipelines.items():
        result = qa(question=question, context=context)
        results[model_name] = result['answer']
    
    return block_name, context, results


def main():
    file_path = "domcasmurro.txt"
    text = load_text(file_path)
    
    chunks = split_into_chapters(text)
    print(f"Número de blocos/capítulos: {len(chunks)}")
    
    embed_model_name = "all-MiniLM-L6-v2"
    print(f"Carregando modelo de embeddings: {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)
    
    names, embeddings, texts_blocks = create_embeddings(chunks, embed_model)
    
    index = build_index(embeddings)
    print("Índice criado com sucesso.")
    
    print("Carregando pipelines de Question Answering...")
    qa_pipelines = {
        "distilbert-base-uncased-distilled-squad": pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
        "deepset/roberta-base-squad2": pipeline("question-answering", model="deepset/roberta-base-squad2")
    }
    
    test_questions = [
        "Quem é Capitu?",
        "Por que Bentinho deveria ir para o seminário?",
        "Quem era José Dias?",
        "O que motivou a construção da casa no Engenho Novo?"
    ]
    
    for question in test_questions:
        print("\n====================================")
        print(f"Pergunta: {question}")
        block_name, context, answers = answer_question(question, embed_model, index, names, texts_blocks, qa_pipelines)
        print(f"Contexto recuperado: {block_name}\n")
        for model_name, answer in answers.items():
            print(f"Resposta ({model_name}): {answer}")
    print("\nExecução concluída.")


if __name__ == "__main__":
    main()

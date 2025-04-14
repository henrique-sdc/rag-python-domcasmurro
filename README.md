# Sistema RAG para Dom Casmurro

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-Usado-yellow.svg)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Usado-orange.svg)
![HNSWLib](https://img.shields.io/badge/hnswlib-Usado-lightgrey.svg)
![NumPy](https://img.shields.io/badge/NumPy-Usado-blueviolet.svg)
![NLTK](https://img.shields.io/badge/NLTK-Opcional-green.svg)

## 📌 Visão Geral

Este projeto implementa um sistema de **Retrieval-Augmented Generation (RAG)** para responder perguntas sobre o romance clássico "Dom Casmurro" de Machado de Assis. O objetivo é demonstrar na prática o fluxo completo de um sistema RAG simples:

1.  **Pré-processamento:** Carregar e dividir o texto do livro em blocos (capítulos ou trechos).
2.  **Geração de Embeddings:** Criar representações vetoriais (embeddings) para cada bloco de texto usando `SentenceTransformers`.
3.  **Indexação:** Construir um índice de busca vetorial eficiente com `hnswlib` para encontrar rapidamente os blocos mais relevantes.
4.  **Recuperação:** Dada uma pergunta, gerar seu embedding e buscar o bloco de texto mais similar no índice.
5.  **Geração de Resposta:** Utilizar modelos de Question Answering (QA) da biblioteca `transformers` (Hugging Face) para gerar uma resposta baseada na pergunta e no bloco de texto recuperado como contexto.

## 🛠️ Tecnologias Utilizadas

-   **Python 3.x**
-   **sentence-transformers:** Para gerar embeddings semânticos (`all-MiniLM-L6-v2`).
-   **transformers (Hugging Face):** Para os pipelines de Question Answering (`distilbert-base-uncased-distilled-squad` e `deepset/roberta-base-squad2`).
-   **hnswlib:** Para criar e consultar o índice de busca vetorial por similaridade (ANN - Approximate Nearest Neighbors).
-   **numpy:** Para manipulação eficiente de arrays numéricos (embeddings).
-   **nltk (opcional):** Utilizado implicitamente ou para tokenização, se necessário.
-   **re:** Para expressões regulares na divisão do texto.

## 📋 Pré-requisitos

Antes de executar o projeto, certifique-se de ter:

-   **Python 3.x** instalado.
-   **pip** (gerenciador de pacotes Python).
-   **Git** (para clonar o repositório).
-   **(Apenas Windows) Microsoft C++ Build Tools:** A biblioteca `hnswlib` requer compilação C++. Se você estiver no Windows, instale as "Build Tools for Visual Studio" a partir [deste link](https://visualstudio.microsoft.com/visual-cpp-build-tools/), certificando-se de selecionar a carga de trabalho "Desenvolvimento para desktop com C++".

## 📂 Arquivos no Repositório

```
rag-python-domcasmurro/
├── domcasmurro.txt # Texto completo do livro
├── rag_domcasmurro.py # Script principal do sistema RAG
└── README.md # Este arquivo
```

## ⚙️ Configuração e Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/henrique-sdc/rag-python-domcasmurro.git
    cd rag-python-domcasmurro
    ```

2.  **Instale as dependências:**
    ```bash
    pip install sentence-transformers transformers hnswlib numpy nltk
    ```
    *(Nota: `torch` ou `tensorflow` geralmente são necessários para `transformers` e `sentence-transformers`)*

3.  **(Opcional/Primeira vez) Baixar dados do NLTK:**
    Se encontrar erros relacionados ao NLTK (como 'punkt'), execute um interpretador Python e rode:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## ▶️ Executando o Sistema

Após a configuração, execute o script Python diretamente do terminal:

   ```bash
    python rag_domcasmurro.py
   ```
O script irá:

1. Carregar domcasmurro.txt.
2. Dividir o texto em capítulos/blocos.
3. Baixar e carregar os modelos de embedding e QA (pode levar um tempo na primeira execução).
4. Gerar e indexar os embeddings.
5. Processar as perguntas de teste predefinidas.
6. Imprimir a pergunta, o capítulo/bloco recuperado como contexto e as respostas geradas por cada modelo de QA.

## 🚀 Exemplo de Saída Esperada

```
====================================
Pergunta: Quem é Capitu?
Contexto recuperado: Capítulo X [Título do Capítulo]

Resposta (distilbert-base-uncased-distilled-squad): [Resposta gerada pelo modelo DistilBERT]
Resposta (deepset/roberta-base-squad2): [Resposta gerada pelo modelo RoBERTa]

====================================
Pergunta: Por que Bentinho deveria ir para o seminário?
Contexto recuperado: Capítulo Y [Título do Capítulo]

Resposta (distilbert-base-uncased-distilled-squad): [Resposta gerada pelo modelo DistilBERT]
Resposta (deepset/roberta-base-squad2): [Resposta gerada pelo modelo RoBERTa]

... (e assim por diante para as outras perguntas) ...

Execução concluída.
```
*(Nota: A qualidade e exatidão das respostas dependem do bloco recuperado e da capacidade do modelo de QA.)*

## 📜 Modelos Utilizados

-   **Embedding Model:** `all-MiniLM-L6-v2` (de `sentence-transformers`)
-   **QA Models:**
    -   `distilbert-base-uncased-distilled-squad`
    -   `deepset/roberta-base-squad2` (ambos de `transformers`)

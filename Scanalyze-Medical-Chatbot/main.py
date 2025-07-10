import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from fastapi import FastAPI
from transformers import pipeline
from langdetect import detect

# Updated imports to avoid deprecation warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# Azure ML 
from azureml.core import Workspace, Model


# Initialize FastAPI app
app = FastAPI()

# Load and configure the model using Unsloth (4-bit QLoRA)
model_name = "FreedomIntelligence/Apollo-7B"

# Load workspace and model
ws = Workspace.from_config()  # Requires config.json in same folder
model = Model(workspace=ws, name="apollo-7b")  # Registered model name
model_path = model.download(target_dir="./model", exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

# Initialize the HuggingFace text-generation pipeline
llm_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    # max_new_tokens=200,
    # temperature=0.3,
    # top_k=30,
    # top_p=0.7,
    # do_sample=True,
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Initialize embeddings and vector store
embedding = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1")

qdrant_client = QdrantClient(
    url="https://12efeef2-9f10-4402-9deb-f070977ddfc8.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Jb39rYQW2rSE9RdXrjdzKY6T1RF44XjdQzCvzFkjat4",
)

qdrant_vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="arabic_rag_collection",
    embeddings=embedding
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=qdrant_vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# Prompt generator based on language
def generate_prompt(question: str) -> str:
    lang = detect(question)

    if lang == "ar":
        return f"""أجب على السؤال الطبي التالي بعربية فصحى واضحة ومفصلة. اعتمد على السياق المتاح، وإن لم يكن كافياً فاستخدم خبرتك الطبية.

التزم بالمعايير التالية:
- إجابة مباشرة بدون تكرار
- أسلوب واضح ومتدفق
- تجنب الحشو غير المفيد

السؤال: {question}
الإجابة:
"""
    else:
        return f"""Answer the following medical question in clear, formal English using a combination of any provided context and your prior medical knowledge.

Instructions:
- Do not repeat phrases or restate the question.
- If context is insufficient, use your general medical understanding to complete the answer.
- Keep the answer informative, structured, and free of unnecessary filler.
- If multiple points are needed, list them clearly and concisely.

Question: {question}
Answer:"""


def clean_response(response):
    """Remove repetitive or near-repetitive sentences from Arabic/English response"""
    # Use regex to split Arabic or English sentences
    sentences = re.split(r'[.!؟]\s*', response)
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check for exact or near-duplicate sentences
        is_repetitive = False
        for seen in seen_sentences:
            if len(sentence) > 10 and len(seen) > 10:
                sentence_words = set(sentence.lower().split())
                seen_words = set(seen.lower().split())
                if len(sentence_words & seen_words) / max(len(sentence_words), len(seen_words)) > 0.7:
                    is_repetitive = True
                    break

        if not is_repetitive:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence)

    # Reconstruct with proper punctuation
    return '، '.join(unique_sentences).strip() + '.' if unique_sentences else response


# Root endpoint
@app.get("/")
def root():
    return {"message": "Hello from Scanalyze!"}

# Question answering endpoint
@app.post("/ask")
def chat_fn(message: str):
    prompt = generate_prompt(message)

    response = qa_chain.run(prompt)
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    elif "الإجابة:" in response:
        answer = response.split("الإجابة:")[-1].strip()

    # answer = clean_response(answer)
    # answer = clean_arabic_text(answer)
    # answer = truncate_at_sentence_end(answer)

    return {'response': response, "Answer": answer}
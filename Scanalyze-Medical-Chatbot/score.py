import torch
import re
from langdetect import detect
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Globals
qa_chain = None

def init():
    global qa_chain

    model_name = "FreedomIntelligence/Apollo-7B"

    # Load workspace and model
    ws = Workspace.from_config()  # Requires config.json in same folder
    model = Model(workspace=ws, name="apollo-7b")  # Registered model name
    model_path = model.download(target_dir="./model", exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token

    # Pipeline
    llm_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Embeddings and vector store
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

    qa_chain_local = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=qdrant_vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )

    qa_chain = qa_chain_local
    print("Model and QA chain initialized.")

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

def run(input_data):
    try:
        global qa_chain
        message = input_data.get("message", "")
        if not message:
            return {"error": "Missing 'message'"}

        prompt = generate_prompt(message)
        response = qa_chain.run(prompt)

        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        elif "الإجابة:" in response:
            answer = response.split("الإجابة:")[-1].strip()
        else:
            answer = response

        return {"response": response, "Answer": answer}

    except Exception as e:
        return {"error": str(e)}

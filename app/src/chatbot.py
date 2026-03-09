from typing import List
import pymupdf
import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from elevenlabs import ElevenLabs
from pathlib import Path
import requests
import json

load_dotenv()

# ────────────────────────────────────────────────
# Clients
# ────────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
PINECONE_INDEX_NAME = "chatbot-knowledge-base"
PINECONE_DIMENSION = 1536
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 80
TOP_K = 5

GENERATION_MODEL = "gpt-4o-mini"
TEXT_TO_SPEECH_MODEL = "gpt-4o-mini-tts"
VOICE = "coral"
VOICE_INSTRUCTIONS = "Speak clearly in a helpful assistant tone."

#TRANSCRIPTION_MODEL="scribe_v1"
TRANSCRIPTION_MODEL="whisper-1"
URL_SPEECH_TO_TEXT="https://api.elevenlabs.io/v1/speech-to-text"

CHATBOT_TEMPERATURE = 0.25
CHATBOT_MAX_TOKENS = 200
CHATBOT_PROMPT = """
You are a helpful assistant that answers questions using the provided context
and conversation history.

Rules:
- Use the retrieved context as the main source of factual information.
- You may also use the conversation history to understand previous questions.
- If the user asks about previous conversation, answer using the chat history.
- If the answer is not found in either the context or chat history, respond with:
"The information is out of my knowledge base."
"""

CHAT_HISTORY_FILE = "chat_history.json"

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def generate_pdf_id_and_name(input_file: str) -> tuple[str, str]:
    pdf_id = str(uuid.uuid4())
    pdf_name = os.path.basename(input_file)
    return pdf_id, pdf_name


def word_extractor(input_file: str) -> str:
    text = ""
    with pymupdf.open(input_file) as doc:
        for page in doc:
            text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def create_pinecone_index() -> None:
    existing = {idx.name for idx in pc.list_indexes()}

    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating index: {PINECONE_INDEX_NAME}")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )


def chunk_and_embed(text: str, pdf_name: str, pdf_id: str) -> None:
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    index = pc.Index(PINECONE_INDEX_NAME)
    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunk
        ).data[0].embedding

        vectors.append({
            "id": f"{pdf_id}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "chunk_id": i,
                "text": chunk
            }
        })

    index.upsert(vectors=vectors)
    print(f"Uploaded {len(vectors)} chunks to Pinecone")

def transcribe(input_file: str) -> str:
    with open(input_file, "rb") as audio_file:
        response = requests.post(
            URL_SPEECH_TO_TEXT,
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            data={"model_id": TRANSCRIPTION_MODEL},
            files={"file": (os.path.basename(input_file), audio_file)}
        )

    result = response.json()
    print(result)  # Debug: see actual API response

    return result.get("text", "")


def transcribe(input_file: str) -> str:
    with open(input_file, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audio_file,
            response_format="text", # or 'json', 'srt', 'verbose_json'
            language="en" # optional, auto-detect if omitted
    )
    return transcript

def load_chat_history(history_file):

    if not os.path.exists(history_file):
        return {"conversation": []}

    with open(history_file, "r") as f:
        return json.load(f)

def save_chat_history(chat_history, history_file):

    with open(history_file, "w") as f:
        json.dump(chat_history, f, indent=2)

def format_chat_history(history, limit=3):

    recent_history = history[-limit:]

    formatted = ""

    for chat in recent_history:
        formatted += f"User: {chat['user']}\n"
        formatted += f"Assistant: {chat['assistant']}\n\n"

    return formatted

def build_prompt_context(chunks, history):

    context = "\n\n".join(chunks)

    history_text = format_chat_history(history)

    return history_text, context

def text_to_speech(text: str, output_file: str) -> bytes:
    """
    Convert text to speech using OpenAI and save as mp3.
    """

    speech_file_path = Path(output_file)

    with client.audio.speech.with_streaming_response.create(
        model=TEXT_TO_SPEECH_MODEL,
        voice= VOICE,
        input=text,
        instructions= VOICE_INSTRUCTIONS
    ) as response:

        response.stream_to_file(speech_file_path)

    print(f"Speech saved to {speech_file_path}")

    with open(speech_file_path, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes

def embed_and_search_chunks(transcription: str, top_k: int = TOP_K):
    index = pc.Index(PINECONE_INDEX_NAME)

    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=transcription
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    chunks = [match["metadata"]["text"] for match in results["matches"]]

    return chunks

def build_context(chunks: list[str]) -> str:
    """
    Combine retrieved chunks into a single context string.
    """
    context = "\n\n".join(chunks)
    return context

def generate_llm_response(query: str, context: str, history: str) -> str:

    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
    {"role": "system", "content": CHATBOT_PROMPT},

    {"role": "system", "content": f"Conversation History:\n{history}"},

    {"role": "system", "content": f"Context:\n{context}"},

    {"role": "user", "content": query}
],
        temperature=CHATBOT_TEMPERATURE,
        max_tokens=CHATBOT_MAX_TOKENS
    )

    return resp.choices[0].message.content.strip()


def delete_vectors_by_filter(pdf_name: str = None, pdf_id: str = None) -> None:
    index = pc.Index(PINECONE_INDEX_NAME)

    if not pdf_name and not pdf_id:
        raise ValueError("You must provide pdf_name or pdf_id")

    filter_query = {}

    if pdf_id:
        filter_query["pdf_id"] = {"$eq": pdf_id}

    if pdf_name:
        filter_query["pdf_name"] = {"$eq": pdf_name}

    print(f"Deleting vectors with filter: {filter_query}")

    index.delete(filter=filter_query)

    print("Deletion request sent to Pinecone.")
# ────────────────────────────────────────────────
# Main Ingestion Pipeline
# ────────────────────────────────────────────────
def add_pdf_to_knowledge_base(input_file: str) -> None:
    pdf_id, pdf_name = generate_pdf_id_and_name(input_file)
    text = word_extractor(input_file)
    create_pinecone_index()
    chunk_and_embed(text, pdf_name, pdf_id)
    print (f"PDF '{pdf_name}' added to knowledge base with ID: {pdf_id}")
    return pdf_id, pdf_name


def talk_with_pdf(input_audio: str, output_dir: str, chat_history_file: str):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define audio output path
    audio_output_path = os.path.join(output_dir, "response.mp3")

    # Load chat history JSON
    chat_history = load_chat_history(chat_history_file)

    history = chat_history.get("conversation", [])

    # Transcribe user audio
    query = transcribe(input_audio)

    # Retrieve relevant chunks
    chunks = embed_and_search_chunks(query)

    # Build context
    context = build_context(chunks)

    # Format history for LLM
    history_text = format_chat_history(history)

    # Generate response
    text = generate_llm_response(query, context, history_text)

    # Convert to speech
    audio = text_to_speech(text, audio_output_path)

    # Update history
    history.append({
        "user": query.strip(),
        "assistant": text.strip()
    })

    chat_history["conversation"] = history

    # Save updated history
    save_chat_history(chat_history, chat_history_file)

    print(f"Chatbot Response: {text}")

    return audio, text


def delete_pdf_from_knowledge_base(pdf_name: str = None, pdf_id: str = None) -> None:
    delete_vectors_by_filter(pdf_name, pdf_id)

# ────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # input_file = r"..\files\Notes2.pdf"
    # add_pdf_to_knowledge_base(input_file)
    
    # input_files = r"..\files\Recording.mp3"
    # response = transcribe(input_files)
    # print(response)
    # input_files = r"..\files\Recording.mp3"
    # response = chat_with_pdf(input_files)
    # print(f"Chatbot Response: {response}")

    # input_audio = r"C:\files\scott\app\files\Django.mp3"
    # output_dir = r"C:\files\scott\app\output"
    # response = talk_with_pdf(input_audio, output_dir)
    # print(f"Chatbot Response generated and saved.")

    # transcription = "What is Base Template in Django?"
    # chunks = embed_and_search_chunks(transcription)
    # print("Top relevant chunks:")
    # for chunk in chunks:
    #     print(f"- {chunk}")

    # query = "What is supervised learning?"
    # response = chat_with_pdf(query)
    # print(f"Chatbot Response: {response}")

    # pdf_name = "Notes2.pdf"
    # pdf_id = "4e570231-ad55-4c66-9496-27eedd52ba3c"
    # delete_pdf_from_knowledge_base(pdf_name=pdf_name, pdf_id=pdf_id)
    # print(f"Deleted PDF from knowledge base: {pdf_name} (ID: {pdf_id})")


    input_audio = r"C:\files\scott\app\files\context.mp3"
    output_dir = r"C:\files\scott\app\output"
    chat_history = r"C:\files\scott\app\output\chat_history.json"

    audio, text = talk_with_pdf(
        input_audio,
        output_dir,
        chat_history
    )

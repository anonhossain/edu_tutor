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

# voice_id = "pNInz6obpgDQGcFmaJgB"

# def text_to_speech(text: str, output_file: str) -> bytes:
#     """
#     Convert text to speech and save it as an mp3 file.
#     """

#     client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

#     response_stream = client.text_to_speech.convert(
#         voice_id=voice_id,
#         model_id="eleven_multilingual_v2",
#         text=text,
#         output_format="mp3_44100_128",
#         voice_settings={
#             "stability": 0.5,
#             "similarity_boost": 0.9,
#             "style": 1.0,
#             "speed": 0.75,
#             "use_speaker_boost": True
#         }
#     )

#     audio_chunks = []

#     for chunk in response_stream:
#         if chunk:
#             audio_chunks.append(chunk)

#     audio_bytes = b"".join(audio_chunks)

#     with open(output_file, "wb") as f:
#         f.write(audio_bytes)

#     print(f"Speech generation complete. Audio length: {len(audio_bytes)} bytes")

#     return audio_bytes


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

import requests

def generate_llm_response(query: str, context: str) -> str:
    """
    Generate response from LLM using context and user query.
    """

    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer questions using the provided context. "
                "If the answer is not in the context, say: The informaition is out of my knowledge base."
            },
            {
                "role": "user",
                "content": f"""
                    Context:
                    {context}

                    Question:
                    {query}

                    Answer:
                    """
            }
        ],
        temperature=0.25,
        max_tokens=300
    )

    content = resp.choices[0].message.content.strip()
    return content

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

# def chat_with_pdf(input_file: str) -> str:
#     query = transcribe(input_file)
#     chunks = embed_and_search_chunks(query)
#     context = build_context(chunks)
#     text = generate_llm_response(query, context)
#     response = text_to_speech(text, "response.mp3")
#     return response


def chat_with_pdf(input_file: str):

    query = transcribe(input_file)

    chunks = embed_and_search_chunks(query)

    context = build_context(chunks)

    text = generate_llm_response(query, context)

    audio = text_to_speech(text, "response.mp3")

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

    input_files = r"..\files\Recording.mp3"
    response = chat_with_pdf(input_files)
    print(f"Chatbot Response generated and saved.")

    # transcription = "What is Base Template in Django?"
    # chunks = embed_and_search_chunks(transcription)
    # print("Top relevant chunks:")
    # for chunk in chunks:
    #     print(f"- {chunk}")

    # query = "What is supervised learning?"
    # response = chat_with_pdf(query)
    # print(f"Chatbot Response: {response}")

    # pdf_name = "Notes2.pdf"
    # pdf_id = "dd4533b3-4e72-4d43-9f22-73e8bad8066a"
    # delete_pdf_from_knowledge_base(pdf_name=pdf_name, pdf_id=pdf_id)
    # print(f"Deleted PDF from knowledge base: {pdf_name} (ID: {pdf_id})")
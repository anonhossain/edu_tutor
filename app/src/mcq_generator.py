import os
import json
from typing import List

import pymupdf  # fitz
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field

load_dotenv()

# ────────────────────────────────────────────────
# Clients
# ────────────────────────────────────────────────

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ────────────────────────────────────────────────
# CONFIGURATION ─ all important tunable values in one place
# ────────────────────────────────────────────────

# Pinecone settings
PINECONE_INDEX_NAME      = "mcq-generator"
PINECONE_DIMENSION       = 1536
PINECONE_METRIC          = "cosine"
PINECONE_CLOUD           = "aws"
PINECONE_REGION          = "us-east-1"
PINECONE_TOP_K           = 10                  # how many chunks to retrieve

# Embedding model
EMBEDDING_MODEL          = "text-embedding-3-small"

# Chunking settings
CHUNK_SIZE               = 1000
CHUNK_OVERLAP            = 150

# LLM settings - topic extraction
TOPIC_EXTRACTION_MODEL   = "gpt-4o-mini"
TOPIC_EXTRACTION_TEMP    = 0.2
TOPIC_EXTRACTION_MAX_TOK = 8192                  # generous, but safe

# LLM settings - MCQ generation per topic
MCQ_GENERATION_MODEL     = "gpt-4o-mini"
MCQ_GENERATION_TEMP      = 0.25
MCQ_GENERATION_MAX_TOK   = 4096

# General generation settings
DEFAULT_TARGET_MCQ       = 50                    # fallback if not passed
CORPUS_SAMPLE_LIMIT      = 28000                 # chars for MCQ prompt
TOPIC_CORPUS_LIMIT       = 32000                 # chars for topic extraction

# Output
OUTPUT_FILENAME          = "generated_mcqs.json"


# ────────────────────────────────────────────────
# Pydantic Models
# ────────────────────────────────────────────────

class MCQ(BaseModel):
    mcq_no: int
    topic: str = Field(..., description="Main topic this question belongs to")
    question: str
    option_a: str = Field(..., alias="option_a")
    option_b: str = Field(..., alias="option_b")
    option_c: str = Field(..., alias="option_c")
    option_d: str = Field(..., alias="option_d")
    correct_ans: str = Field(..., description="Letter only: a, b, c or d")
    explanation: str

    class Config:
        populate_by_name = True


class MCQList(BaseModel):
    mcqs: List[MCQ]


class Topic(BaseModel):
    name: str
    summary: str


class TopicList(BaseModel):
    topics: List[Topic]


# ────────────────────────────────────────────────
# PDF → Text
# ────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from PDF file."""
    text = ""
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text.strip()


# ────────────────────────────────────────────────
# Text → Chunks
# ────────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    """Simple overlapping chunker using global config values."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ────────────────────────────────────────────────
# Chunk + Embed + Upsert to Pinecone
# ────────────────────────────────────────────────

def index_document_in_pinecone(text: str):
    chunks = chunk_text(text)

    # Create index if missing
    existing = {idx.name for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    vectors = []

    for i, chunk in enumerate(chunks):
        try:
            emb = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=chunk,
                encoding_format="float"
            ).data[0].embedding

            vectors.append({
                "id": f"chunk_{i}",
                "values": emb,
                "metadata": {"text": chunk}
            })
        except Exception as e:
            print(f"Embedding failed for chunk {i}: {e}")

    if vectors:
        index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} chunks into Pinecone.")
    else:
        print("No vectors to upsert.")


# ────────────────────────────────────────────────
# Retrieve all chunks (dummy vector query)
# ────────────────────────────────────────────────

def retrieve_all_text() -> str:
    index = pc.Index(PINECONE_INDEX_NAME)
    dummy_vector = [0.0] * PINECONE_DIMENSION

    try:
        res = index.query(
            vector=dummy_vector,
            top_k=PINECONE_TOP_K,
            include_metadata=True,
            include_values=False
        )
        texts = [
            match["metadata"]["text"]
            for match in res["matches"]
            if "text" in match["metadata"]
        ]
        return "\n\n".join(texts)
    except Exception as e:
        print(f"Pinecone query failed: {e}")
        return ""


# ────────────────────────────────────────────────
# Generate MCQs using structured outputs
# ────────────────────────────────────────────────

def generate_mcqs_for_topic(
    topic: Topic,
    count: int,
    corpus_sample: str,
) -> List[MCQ]:
    system_prompt = (
        "You are an expert MCQ creator for educational content. "
        "Generate high-quality, accurate multiple-choice questions. "
        "Always return valid JSON matching the requested schema. "
        "Correct answer must be one of: a, b, c, d"
    )

    user_prompt = f"""\
Study material excerpt:
{corpus_sample[:CORPUS_SAMPLE_LIMIT]}

Topic: {topic.name}
Summary: {topic.summary}

Generate exactly {count} diverse, non-duplicate MCQs on this topic.
Return ONLY valid JSON in this exact structure:

{{
  "mcqs": [
    {{
      "mcq_no": 1,
      "topic": "{topic.name}",
      "question": "...",
      "option_a": "...",
      "option_b": "...",
      "option_c": "...",
      "option_d": "...",
      "correct_ans": "a",
      "explanation": "..."
    }}
  ]
}}
"""

    try:
        completion = client.beta.chat.completions.parse(
            model=MCQ_GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=MCQList,
            temperature=MCQ_GENERATION_TEMP,
            max_tokens=MCQ_GENERATION_MAX_TOK
        )

        parsed = completion.choices[0].message.parsed
        if parsed and parsed.mcqs:
            return parsed.mcqs
        else:
            print("No MCQs parsed from response.")
            return []

    except Exception as e:
        print(f"MCQ generation failed for topic '{topic.name}': {e}")
        return []


# ────────────────────────────────────────────────
# Main MCQ generation logic
# ────────────────────────────────────────────────

def generate_mcqs(mcq_target: int = DEFAULT_TARGET_MCQ) -> MCQList:
    corpus = retrieve_all_text()

    if not corpus:
        raise ValueError("No text retrieved from Pinecone. Cannot generate MCQs.")

    # ── Extract main topics ───────────────────────────────────────
    topic_extraction_prompt = f"""\
From the following study material, identify the most important learning topics.

Return ONLY valid JSON:

{{
  "topics": [
    {{"name": "Topic Name", "summary": "One-sentence summary"}}
  ]
}}

Study material:
{corpus[:TOPIC_CORPUS_LIMIT]}
"""

    try:
        topic_completion = client.beta.chat.completions.parse(
            model=TOPIC_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You return only structured JSON."},
                {"role": "user",   "content": topic_extraction_prompt}
            ],
            response_format=TopicList,
            temperature=TOPIC_EXTRACTION_TEMP,
            max_tokens=TOPIC_EXTRACTION_MAX_TOK
        )

        topics = topic_completion.choices[0].message.parsed.topics

    except Exception as e:
        print(f"Topic extraction failed: {e}")
        return MCQList(mcqs=[])

    if not topics:
        print("No topics extracted.")
        return MCQList(mcqs=[])

    print(f"Found {len(topics)} topics.")

    # Distribute MCQs fairly
    base = mcq_target // len(topics)
    extra = mcq_target % len(topics)

    all_mcqs: List[MCQ] = []
    seen_questions = set()
    global_counter = 1

    for i, topic in enumerate(topics):
        num = base + (1 if i < extra else 0)
        if num == 0:
            continue

        print(f"→ Generating {num} MCQs for: {topic.name}")

        generated = generate_mcqs_for_topic(
            topic=topic,
            count=num,
            corpus_sample=corpus
        )

        for mcq in generated:
            q_clean = mcq.question.strip().lower()
            if q_clean and q_clean not in seen_questions:
                seen_questions.add(q_clean)
                mcq.mcq_no = global_counter
                global_counter += 1
                all_mcqs.append(mcq)

    print(f"Total unique MCQs generated: {len(all_mcqs)}")
    return MCQList(mcqs=all_mcqs)


# ────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────

def run_pipeline(
    pdf_path: str,
    output_dir: str,
    target_mcqs: int
):
    os.makedirs(output_dir, exist_ok=True)

    print("1. Extracting text from PDF...")
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        print("No text extracted. Exiting.")
        return

    print("2. Chunking & embedding to Pinecone...")
    index_document_in_pinecone(full_text)

    print("3. Generating MCQs...")
    mcq_list = generate_mcqs(target_mcqs)

    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mcq_list.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Done. Saved {len(mcq_list.mcqs)} MCQs → {output_path}")

    # Optional cleanup
    try:
        pc.delete_index(PINECONE_INDEX_NAME)
        print(f"Index '{PINECONE_INDEX_NAME}' removed successfully.")
    except Exception as e:
        print(f"Could not delete index: {e}")


if __name__ == "__main__":
    PDF_FILE = "app/files/Notes2.pdf"
    OUTPUT_FOLDER = "app/output"

    run_pipeline(PDF_FILE, OUTPUT_FOLDER, target_mcqs=25)
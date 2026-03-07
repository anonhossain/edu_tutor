# import os
# import pymupdf
# from pinecone import Pinecone, ServerlessSpec
# from openai import OpenAI
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from typing import List, Dict
# from pinecone import Pinecone
# from openai import OpenAI
# import os
# import json
# from pinecone import Pinecone
# import os


# load_dotenv()

# pinecone_api = os.getenv("PINECONE_API_KEY")
# openai_api = os.getenv("OPENAI_API_KEY")
# # Initialize clients
# pc = Pinecone(api_key=pinecone_api)
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# client = OpenAI(api_key=openai_api)
# client = OpenAI()
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# # ---------- Pydantic Models ----------

# class Topic(BaseModel):
#     topic_title: str
#     content: str


# class Chapter(BaseModel):
#     chapter_title: str
#     topics: List[Topic]


# class Curriculum(BaseModel):
#     chapters: List[Chapter]



# def word_extractor(input_folder, output_folder):
#     """
#     Extract text from all PDFs in a folder and merge them into a single .txt file.
#     The output filename will be the name of the first PDF file.
#     """

#     os.makedirs(output_folder, exist_ok=True)

#     merged_text = ""
#     first_pdf_name = None

#     for file in os.listdir(input_folder):
#         if file.lower().endswith(".pdf"):

#             if first_pdf_name is None:
#                 first_pdf_name = os.path.splitext(file)[0] + ".txt"

#             pdf_path = os.path.join(input_folder, file)

#             try:
#                 doc = pymupdf.open(pdf_path)

#                 merged_text += f"\n\n===== START OF {file} =====\n\n"

#                 for page in doc:
#                     merged_text += page.get_text()

#                 merged_text += f"\n\n===== END OF {file} =====\n\n"

#                 print(f"Processed: {file}")

#             except Exception as e:
#                 print(f"Error processing {file}: {e}")

#     if first_pdf_name is None:
#         print("No PDF files found.")
#         return

#     output_path = os.path.join(output_folder, first_pdf_name)

#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(merged_text)

#     print(f"\nMerged text saved to: {output_path}")



# def chunk_text(text, chunk_size=1000, overlap=100):
#     chunks = []
#     start = 0

#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap

#     return chunks


# def chunk_and_embed(input_text):
#     index_name = "course-curriculum"
#     dimension = 1536
#     metric = "cosine"
#     cloud = "aws"
#     region = "us-east-1"
#     chunk_size = 1000
#     overlap = 100
#     embedding_model = "text-embedding-3-small"

#     # Read txt file
#     with open(input_text, "r", encoding="utf-8") as f:
#         text = f.read()

#     # Chunking
#     chunks = chunk_text(text, chunk_size, overlap)

#     # Create Pinecone index if not exists
#     existing_indexes = [index.name for index in pc.list_indexes()]

#     if index_name not in existing_indexes:
#         pc.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric=metric,
#             spec=ServerlessSpec(
#                 cloud=cloud,
#                 region=region
#             )
#         )

#     index = pc.Index(index_name)

#     vectors = []

#     for i, chunk in enumerate(chunks):

#         embedding = client.embeddings.create(
#             model=embedding_model,
#             input=chunk
#         )

#         vector = {
#             "id": f"chunk_{i}",
#             "values": embedding.data[0].embedding,
#             "metadata": {
#                 "text": chunk
#             }
#         }

#         vectors.append(vector)

#     index.upsert(vectors=vectors)

#     print("Completed chunking and saving to Pinecone")


# # ---------- Curriculum Maker ----------

# def curriculum_maker(output_folder: str):

#     index_name = "course-curriculum"
#     summarization_model = "gpt-4.1-mini"
#     generation_model = "gpt-4.1"

#     os.makedirs(output_folder, exist_ok=True)

#     index = pc.Index(index_name)

#     print("Retrieving chunks from Pinecone...")

#     results = index.query(
#         vector=[0] * 1536,
#         top_k=1000,
#         include_metadata=True
#     )

#     chunks = [match["metadata"]["text"] for match in results["matches"]]

#     print(f"Retrieved {len(chunks)} chunks")

#     # ---------- Step 1: Summarize Chunks ----------

#     summaries = []

#     for chunk in chunks:

#         response = client.chat.completions.create(
#             model=summarization_model,
#             messages=[
#                 {"role": "system", "content": "Summarize the text and extract key concepts."},
#                 {"role": "user", "content": chunk}
#             ]
#         )

#         summaries.append(response.choices[0].message.content)

#     print("Chunk summarization completed")

#     # ---------- Step 2: Identify Chapters ----------

#     combined_summary = "\n".join(summaries)

#     response = client.chat.completions.create(
#         model=generation_model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Create 5 structured curriculum chapter titles based on the summaries. Return one chapter per line."
#             },
#             {
#                 "role": "user",
#                 "content": combined_summary
#             }
#         ]
#     )

#     chapters_list = [
#         c.strip("- ").strip()
#         for c in response.choices[0].message.content.split("\n")
#         if c.strip()
#     ]

#     print("Chapters identified")

#     # ---------- Step 3: Assign Chunks to Chapters ----------

#     numbered_chapters = "\n".join(
#         [f"{i+1}. {chapter}" for i, chapter in enumerate(chapters_list)]
#     )

#     chapter_chunks: Dict[str, List[str]] = {chapter: [] for chapter in chapters_list}

#     for chunk in chunks:

#         response = client.chat.completions.create(
#             model=summarization_model,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": f"""
#                         Choose the MOST relevant chapter number for this text.

#                         Chapters:
#                         {numbered_chapters}

#                         Return ONLY the chapter number.
#                         """
#                 },
#                 {
#                     "role": "user",
#                     "content": chunk
#                 }
#             ]
#         )

#         answer = response.choices[0].message.content.strip()

#         try:
#             index_number = int(answer) - 1
#             chapter_name = chapters_list[index_number]
#             chapter_chunks[chapter_name].append(chunk)
#         except:
#             pass

#     print("Chunks assigned to chapters")

#     # ---------- Step 4: Generate Topics + Content ----------

#     final_chapters = []

#     for chapter, texts in chapter_chunks.items():

#         if not texts:
#             continue

#         context = "\n".join(texts[:5])

#         topic_response = client.chat.completions.create(
#             model=generation_model,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": f"Generate 3-5 learning topics for the chapter: {chapter}. Return one topic per line."
#                 },
#                 {
#                     "role": "user",
#                     "content": context
#                 }
#             ]
#         )

#         topics_list = [
#             t.strip("- ").strip()
#             for t in topic_response.choices[0].message.content.split("\n")
#             if t.strip()
#         ]

#         topics = []

#         for topic in topics_list:

#             content_response = client.chat.completions.create(
#                 model=generation_model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": f"Write detailed learning material for the topic: {topic}"
#                     },
#                     {
#                         "role": "user",
#                         "content": context
#                     }
#                 ]
#             )

#             content = content_response.choices[0].message.content

#             topics.append(
#                 Topic(
#                     topic_title=topic,
#                     content=content
#                 )
#             )

#         final_chapters.append(
#             Chapter(
#                 chapter_title=chapter,
#                 topics=topics
#             )
#         )

#     curriculum = Curriculum(chapters=final_chapters)

#     # ---------- Save JSON ----------

#     output_path = os.path.join(output_folder, "curriculum.json")

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(curriculum.model_dump(), f, indent=2)

#     print(f"Curriculum saved to: {output_path}")


# def remove_index(index_name):
#     try:
#         existing_indexes = [i["name"] for i in pc.list_indexes()]

#         if index_name in existing_indexes:
#             pc.delete_index(index_name)
#             print(f"Index '{index_name}' removed successfully.")
#         else:
#             print(f"Index '{index_name}' does not exist.")

#     except Exception as e:
#         print(f"Error removing index: {e}")

# def run_curriculum_maker(input_folder: str, output_folder: str):
#     """
#     Full pipeline:
#     1. Extract text from PDFs
#     2. Chunk + Embed into Pinecone
#     3. Generate Curriculum JSON
#     """

#     print("\n========== STEP 1: Extracting Text ==========")

#     word_extractor(input_folder, output_folder)

#     # Find the generated txt file
#     txt_files = [f for f in os.listdir(output_folder) if f.endswith(".txt")]

#     if not txt_files:
#         raise Exception("No text file generated from PDFs.")

#     input_text = os.path.join(output_folder, txt_files[0])

#     print(f"Using extracted file: {input_text}")

#     print("\n========== STEP 2: Chunking + Embedding ==========")

#     chunk_and_embed(input_text)

#     print("\n========== STEP 3: Generating Curriculum ==========")

#     curriculum_maker(output_folder)

#     print("\n========== PIPELINE COMPLETED ==========")


# if __name__ == "__main__":
#     input_folder = "app/files"
#     output_folder = "app/output"
#     run_curriculum_maker(input_folder, output_folder)

#     index_name = "course-curriculum"
#     dimension = 1536
#     metric = "cosine"
#     cloud = "aws"
#     region = "us-east-1"
#     chunk_size = 1000
#     overlap = 100
#     embedding_model = "text-embedding-3-small"

#     input_folder = "app/files"
#     output_folder = "app/output"
#     input_text = "app/output/notes.txt"
    
    
#     # Word Extractor

#     #word_extractor(input_folder, output_folder)

#     # Chunking and Embedding
  
#     #chunk_and_embed(input_text)

#     # Curriculum Maker
#     # curriculum = curriculum_maker()
#     # print(curriculum)

#     # Save Curriculum
#     # curriculum_maker(output_folder)
#     # print("Curriculum generation completed")

#     # Remove Pinecone Index
#     # index_name = "course-curriculum"
#     # remove_index(index_name)
#     # print("index deleted")







import os
import json
from typing import List, Dict

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel


# ────────────────────────────────────────────────
# Load environment variables
# ────────────────────────────────────────────────
load_dotenv()


# ────────────────────────────────────────────────
# Configuration ─ all tunable values in one place
# ────────────────────────────────────────────────

PINECONE_INDEX_NAME     = "course-curriculum"
PINECONE_DIMENSION      = 1536
PINECONE_METRIC         = "cosine"
PINECONE_CLOUD          = "aws"
PINECONE_REGION         = "us-east-1"

EMBEDDING_MODEL         = "text-embedding-3-small"
CHUNK_SIZE              = 1000
CHUNK_OVERLAP           = 100

SUMMARIZATION_MODEL     = "gpt-4o-mini"
GENERATION_MODEL        = "gpt-4o"

RETRIEVE_TOP_K          = 5                 # increased back for better coverage

DEFAULT_INPUT_FOLDER    = "app/files"
DEFAULT_OUTPUT_FOLDER   = "app/output"

SUMMARIZATION_PROMPT    = "Summarize the text and extract key concepts."
CHAPTER_TITLE_PROMPT    = "Create important structured curriculum chapter titles based on the summaries. Return one chapter per line."


# ────────────────────────────────────────────────
# Clients
# ────────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ────────────────────────────────────────────────
# Pydantic Models
# ────────────────────────────────────────────────

class Topic(BaseModel):
    topic_title: str
    content: str


class Chapter(BaseModel):
    chapter_title: str
    topics: List[Topic]


class Curriculum(BaseModel):
    chapters: List[Chapter]


# ────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────

def word_extractor(input_folder: str, output_folder: str | None = None) -> str:
    """
    Extract and merge text from all PDFs in the input folder.
    
    Args:
        input_folder: Folder containing PDF files
        output_folder: Optional - if provided, saves merged text as .txt
    
    Returns:
        str: The complete merged text from all PDFs
    """
    merged_text = ""
    first_pdf_name = None

    for file in os.listdir(input_folder):
        if file.lower().endswith(".pdf"):
            if first_pdf_name is None:
                first_pdf_name = os.path.splitext(file)[0] + ".txt"

            pdf_path = os.path.join(input_folder, file)

            try:
                doc = pymupdf.open(pdf_path)
                merged_text += f"\n\n===== START OF {file} =====\n\n"

                for page in doc:
                    merged_text += page.get_text()

                merged_text += f"\n\n===== END OF {file} =====\n\n"

                print(f"Processed: {file}")
                doc.close()

            except Exception as e:
                print(f"Error processing {file}: {e}")

    if not merged_text.strip():
        print("No PDF files found or no text extracted.")
        return ""

    # Optional: save to file if output_folder is provided
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, first_pdf_name or "merged_notes.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(merged_text)
        print(f"Merged text also saved to: {output_path}")

    return merged_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_and_embed(text: str):
    """Chunk the provided text string and embed + upsert to Pinecone"""
    if not text.strip():
        print("No text provided for chunking and embedding.")
        return

    chunks = chunk_text(text)

    # Create index if missing
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
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
            embedding_resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=chunk
            )
            vector = {
                "id": f"chunk_{i}",
                "values": embedding_resp.data[0].embedding,
                "metadata": {"text": chunk}
            }
            vectors.append(vector)
        except Exception as e:
            print(f"Embedding error on chunk {i}: {e}")

    if vectors:
        index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} vectors to Pinecone")
    else:
        print("No vectors created.")


# ────────────────────────────────────────────────
# Core Curriculum Generation Logic
# ────────────────────────────────────────────────

def curriculum_maker(output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    index = pc.Index(PINECONE_INDEX_NAME)

    print("Retrieving chunks from Pinecone...")
    results = index.query(
        vector=[0.0] * PINECONE_DIMENSION,
        top_k=RETRIEVE_TOP_K,
        include_metadata=True
    )

    chunks = [match["metadata"]["text"] for match in results["matches"] if "text" in match["metadata"]]
    print(f"Retrieved {len(chunks)} chunks")

    # Step 1: Summarize chunks
    summaries = []
    for chunk in chunks:
        try:
            response = client.chat.completions.create(
                model=SUMMARIZATION_MODEL,
                messages=[
                    {"role": "system", "content": SUMMARIZATION_PROMPT},
                    {"role": "user",   "content": chunk}
                ],
                temperature=0.3,
                max_tokens=400
            )
            summaries.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Summarization error: {e}")

    print("Chunk summarization completed")

    # Step 2: Generate chapter titles
    combined_summary = "\n".join(summaries)

    chapter_response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": CHAPTER_TITLE_PROMPT},
            {"role": "user",   "content": combined_summary}
        ]
    )

    chapters_list = [
        line.strip("- ").strip()
        for line in chapter_response.choices[0].message.content.split("\n")
        if line.strip() and len(line.strip()) > 5
    ]

    print(f"Identified {len(chapters_list)} chapters")

    # Step 3: Assign chunks to chapters
    numbered_chapters = "\n".join([f"{i+1}. {ch}" for i, ch in enumerate(chapters_list)])
    chapter_chunks: Dict[str, List[str]] = {ch: [] for ch in chapters_list}

    for chunk in chunks:
        try:
            resp = client.chat.completions.create(
                model=SUMMARIZATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"""
Choose the MOST relevant chapter number for this text chunk.

Chapters:
{numbered_chapters}

Return ONLY the number (1-{len(chapters_list)}).
                        """
                    },
                    {"role": "user", "content": chunk[:4000]}
                ],
                temperature=0.1,
                max_tokens=10
            )
            answer = resp.choices[0].message.content.strip()
            try:
                idx = int(answer) - 1
                if 0 <= idx < len(chapters_list):
                    chapter_name = chapters_list[idx]
                    chapter_chunks[chapter_name].append(chunk)
            except ValueError:
                pass
        except Exception:
            pass

    print("Chunks assigned to chapters")

    # Step 4: Generate topics + content
    final_chapters = []

    for chapter_title, texts in chapter_chunks.items():
        if not texts:
            continue

        context = "\n\n".join(texts[:6])  # limit context

        # Generate topics
        topic_resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Generate 3-5 clear learning topics for chapter: {chapter_title}. Return one per line."
                },
                {"role": "user", "content": context}
            ]
        )

        topics_list = [
            t.strip("- ").strip()
            for t in topic_resp.choices[0].message.content.split("\n")
            if t.strip() and len(t.strip()) > 5
        ]

        topics = []
        for topic_title in topics_list:
            try:
                content_resp = client.chat.completions.create(
                    model=GENERATION_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Write detailed educational material for the topic: {topic_title}"
                        },
                        {"role": "user", "content": context}
                    ]
                )
                content = content_resp.choices[0].message.content.strip()
                topics.append(Topic(topic_title=topic_title, content=content))
            except Exception as e:
                print(f"Content generation failed for topic '{topic_title}': {e}")

        final_chapters.append(
            Chapter(chapter_title=chapter_title, topics=topics)
        )

    curriculum = Curriculum(chapters=final_chapters)

    # Save result
    output_path = os.path.join(output_folder, "curriculum.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(curriculum.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Curriculum saved to: {output_path}")


# ────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────

def remove_index(index_name: str = PINECONE_INDEX_NAME):
    try:
        indexes = [idx["name"] for idx in pc.list_indexes()]
        if index_name in indexes:
            pc.delete_index(index_name)
            print(f"Index '{index_name}' removed successfully.")
        else:
            print(f"Index '{index_name}' does not exist.")
    except Exception as e:
        print(f"Error removing index: {e}")


# ────────────────────────────────────────────────
# Main Pipeline
# ────────────────────────────────────────────────

def run_curriculum_maker(input_folder: str = DEFAULT_INPUT_FOLDER, output_folder: str = DEFAULT_OUTPUT_FOLDER):
    """
    Full pipeline:
    1. Extract text from PDFs → returns string
    2. Chunk & embed the text (in memory)
    3. Generate and save curriculum JSON
    4. Clean up Pinecone index
    """
    print("\n" + "="*60)
    print("     COURSE CURRICULUM GENERATION PIPELINE STARTED")
    print("="*60 + "\n")

    # Step 1
    print("STEP 1: Extracting text from PDFs...")
    merged_text = word_extractor(input_folder, output_folder)  # returns text

    if not merged_text.strip():
        print("ERROR: No text could be extracted. Stopping.")
        return

    print(f"Extracted {len(merged_text):,} characters.")

    # Step 2
    print("\nSTEP 2: Chunking & embedding to Pinecone...")
    chunk_and_embed(merged_text)

    # Step 3
    print("\nSTEP 3: Generating structured curriculum...")
    curriculum_maker(output_folder)

    # Cleanup
    print("\nCleaning up Pinecone index...")
    remove_index()

    print("\n" + "="*60)
    print("           PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)


# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────

if __name__ == "__main__":
    run_curriculum_maker()
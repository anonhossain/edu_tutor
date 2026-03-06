import os
import pymupdf
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict
from pinecone import Pinecone
from openai import OpenAI
import os
import json
from pinecone import Pinecone
import os


load_dotenv()

pinecone_api = os.getenv("PINECONE_API_KEY")
openai_api = os.getenv("OPENAI_API_KEY")
# Initialize clients
pc = Pinecone(api_key=pinecone_api)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=openai_api)
client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# ---------- Pydantic Models ----------

class Topic(BaseModel):
    topic_title: str
    content: str


class Chapter(BaseModel):
    chapter_title: str
    topics: List[Topic]


class Curriculum(BaseModel):
    chapters: List[Chapter]



def word_extractor(input_folder, output_folder):
    """
    Extract text from all PDFs in a folder and merge them into a single .txt file.
    The output filename will be the name of the first PDF file.
    """

    os.makedirs(output_folder, exist_ok=True)

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

            except Exception as e:
                print(f"Error processing {file}: {e}")

    if first_pdf_name is None:
        print("No PDF files found.")
        return

    output_path = os.path.join(output_folder, first_pdf_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged_text)

    print(f"\nMerged text saved to: {output_path}")



def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def chunk_and_embed(input_text):
    index_name = "course-curriculum"
    dimension = 1536
    metric = "cosine"
    cloud = "aws"
    region = "us-east-1"
    chunk_size = 1000
    overlap = 100
    embedding_model = "text-embedding-3-small"

    # Read txt file
    with open(input_text, "r", encoding="utf-8") as f:
        text = f.read()

    # Chunking
    chunks = chunk_text(text, chunk_size, overlap)

    # Create Pinecone index if not exists
    existing_indexes = [index.name for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )

    index = pc.Index(index_name)

    vectors = []

    for i, chunk in enumerate(chunks):

        embedding = client.embeddings.create(
            model=embedding_model,
            input=chunk
        )

        vector = {
            "id": f"chunk_{i}",
            "values": embedding.data[0].embedding,
            "metadata": {
                "text": chunk
            }
        }

        vectors.append(vector)

    index.upsert(vectors=vectors)

    print("Completed chunking and saving to Pinecone")


# ---------- Curriculum Maker ----------

def curriculum_maker(output_folder: str):

    index_name = "course-curriculum"
    summarization_model = "gpt-4.1-mini"
    generation_model = "gpt-4.1"

    os.makedirs(output_folder, exist_ok=True)

    index = pc.Index(index_name)

    print("Retrieving chunks from Pinecone...")

    results = index.query(
        vector=[0] * 1536,
        top_k=1000,
        include_metadata=True
    )

    chunks = [match["metadata"]["text"] for match in results["matches"]]

    print(f"Retrieved {len(chunks)} chunks")

    # ---------- Step 1: Summarize Chunks ----------

    summaries = []

    for chunk in chunks:

        response = client.chat.completions.create(
            model=summarization_model,
            messages=[
                {"role": "system", "content": "Summarize the text and extract key concepts."},
                {"role": "user", "content": chunk}
            ]
        )

        summaries.append(response.choices[0].message.content)

    print("Chunk summarization completed")

    # ---------- Step 2: Identify Chapters ----------

    combined_summary = "\n".join(summaries)

    response = client.chat.completions.create(
        model=generation_model,
        messages=[
            {
                "role": "system",
                "content": "Create 5 structured curriculum chapter titles based on the summaries. Return one chapter per line."
            },
            {
                "role": "user",
                "content": combined_summary
            }
        ]
    )

    chapters_list = [
        c.strip("- ").strip()
        for c in response.choices[0].message.content.split("\n")
        if c.strip()
    ]

    print("Chapters identified")

    # ---------- Step 3: Assign Chunks to Chapters ----------

    numbered_chapters = "\n".join(
        [f"{i+1}. {chapter}" for i, chapter in enumerate(chapters_list)]
    )

    chapter_chunks: Dict[str, List[str]] = {chapter: [] for chapter in chapters_list}

    for chunk in chunks:

        response = client.chat.completions.create(
            model=summarization_model,
            messages=[
                {
                    "role": "system",
                    "content": f"""
Choose the MOST relevant chapter number for this text.

Chapters:
{numbered_chapters}

Return ONLY the chapter number.
"""
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ]
        )

        answer = response.choices[0].message.content.strip()

        try:
            index_number = int(answer) - 1
            chapter_name = chapters_list[index_number]
            chapter_chunks[chapter_name].append(chunk)
        except:
            pass

    print("Chunks assigned to chapters")

    # ---------- Step 4: Generate Topics + Content ----------

    final_chapters = []

    for chapter, texts in chapter_chunks.items():

        if not texts:
            continue

        context = "\n".join(texts[:5])

        topic_response = client.chat.completions.create(
            model=generation_model,
            messages=[
                {
                    "role": "system",
                    "content": f"Generate 3-5 learning topics for the chapter: {chapter}. Return one topic per line."
                },
                {
                    "role": "user",
                    "content": context
                }
            ]
        )

        topics_list = [
            t.strip("- ").strip()
            for t in topic_response.choices[0].message.content.split("\n")
            if t.strip()
        ]

        topics = []

        for topic in topics_list:

            content_response = client.chat.completions.create(
                model=generation_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Write detailed learning material for the topic: {topic}"
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            )

            content = content_response.choices[0].message.content

            topics.append(
                Topic(
                    topic_title=topic,
                    content=content
                )
            )

        final_chapters.append(
            Chapter(
                chapter_title=chapter,
                topics=topics
            )
        )

    curriculum = Curriculum(chapters=final_chapters)

    # ---------- Save JSON ----------

    output_path = os.path.join(output_folder, "curriculum.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(curriculum.model_dump(), f, indent=2)

    print(f"Curriculum saved to: {output_path}")


def remove_index(index_name):
    try:
        existing_indexes = [i["name"] for i in pc.list_indexes()]

        if index_name in existing_indexes:
            pc.delete_index(index_name)
            print(f"Index '{index_name}' removed successfully.")
        else:
            print(f"Index '{index_name}' does not exist.")

    except Exception as e:
        print(f"Error removing index: {e}")


if __name__ == "__main__":


    index_name = "course-curriculum"
    dimension = 1536
    metric = "cosine"
    cloud = "aws"
    region = "us-east-1"
    chunk_size = 1000
    overlap = 100
    embedding_model = "text-embedding-3-small"

    input_folder = "app/files"
    output_folder = "app/output"
    input_text = "app/output/notes.txt"
    
    
    # Word Extractor

    

    # word_extractor(input_folder, output_folder)

    # Chunking and Embedding
  
    chunk_and_embed(input_text)

    # Curriculum Maker
    # curriculum = curriculum_maker()
    # print(curriculum)

    # Save Curriculum
    # curriculum_maker(output_folder)
    # print("Curriculum generation completed")

    # Remove Pinecone Index
    # index_name = "course-curriculum"
    # remove_index(index_name)
    # print("index deleted")
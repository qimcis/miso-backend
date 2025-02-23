from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure OpenAI client
try:
    client = openai.OpenAI()  # This will use OPENAI_API_KEY from environment
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    raise

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class Flashcard(BaseModel):
    question: str
    answer: str

class FlashcardsResponse(BaseModel):
    flashcards: List[Flashcard]

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    finally:
        if 'doc' in locals():
            doc.close()

def split_into_chunks(text: str, max_chunk_size: int = 8000) -> List[str]:
    """Split text into manageable chunks."""
    try:
        chunks = []
        current_chunk = ""
        sentences = text.split(". ")

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting text: {str(e)}")

def process_text_chunks(chunks: List[str]) -> List[str]:
    """Process text chunks with OpenAI API."""
    results = []
    
    try:
        for chunk in chunks:
            if not chunk.strip():
                continue

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract and summarize the key concepts, definitions, and important information from this text. Focus on content that would be valuable for creating flashcards. Be thorough but concise."
                    },
                    {
                        "role": "user",
                        "content": chunk
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            if completion.choices[0].message.content:
                results.append(completion.choices[0].message.content)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text with OpenAI: {str(e)}")

def generate_flashcards(content: str) -> FlashcardsResponse:
    """Generate flashcards from processed content."""
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Create 10 high-quality flashcards from the provided content. Each flashcard should have a clear question and comprehensive answer. Format the output exactly as a JSON object with the structure: {\"flashcards\": [{\"question\": \"string\", \"answer\": \"string\"}]}"
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            response_format={"type": "json_object"}
        )
        
        return FlashcardsResponse.parse_raw(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating flashcards: {str(e)}")

@app.post("/upload-pdf", response_model=FlashcardsResponse)
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Upload a PDF file and generate flashcards from its content.
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Read PDF content
        content = await file.read()
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(content)
        
        if len(extracted_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the PDF")

        # Split text into chunks and process
        chunks = split_into_chunks(extracted_text)
        processed_chunks = process_text_chunks(chunks)
        combined_content = "\n\n".join(processed_chunks)

        # Generate flashcards
        flashcards = generate_flashcards(combined_content)
        return flashcards

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
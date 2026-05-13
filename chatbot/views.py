from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

import os

from .rag import (
    initialize_rag,
    search_query,
    ask_llm,
    model
)


# Global runtime objects
index = None
chunks = None


def home(request):

    global index, chunks

    # Create session if not exists
    if not request.session.session_key:
        request.session.create()

    # Initialize chat history
    if "chat_history" not in request.session:
        request.session["chat_history"] = []

    if request.method == "POST":

        # Check if PDF uploaded first
        if index is None or chunks is None:

            return JsonResponse({
                "answer": "No PDF is currently loaded. Please upload a PDF and wait for success confirmation."
            })

        question = request.POST.get("question")

        if not question:

            return JsonResponse({
                "answer": "Please enter a question."
            })

        # Retrieve relevant chunks
        retrieved_chunk = search_query(
            question,
            model,
            index,
            chunks
        )

        if not retrieved_chunk:

            return JsonResponse({
                "answer": "I cannot find this information in the document"
            })

        # Get previous memory
        history = request.session["chat_history"]

        # Convert memory into text
        history_text = ""

        for item in history:

            history_text += (
                f"User: {item['question']}\n"
                f"AI: {item['answer']}\n\n"
            )

        # Build final context
        context = f"""
Previous Conversation:
{history_text}

Relevant Document Context:
{retrieved_chunk}
"""

        # Generate answer
        try:

            answer = ask_llm(
                question,
                context
            )

        except Exception as e:

            print("LLM Error:", e)

            return JsonResponse({
                "answer": "Error generating response."
            })

        # Save into session memory
        history.append({
            "question": question,
            "answer": answer
        })

        request.session["chat_history"] = history

        return JsonResponse({
            "answer": answer
        })

    return render(request, "home.html")


def upload_pdf(request):

    global index, chunks

    if request.method == "POST":

        pdf_file = request.FILES.get("pdf")

        if not pdf_file:

            return JsonResponse({
                "message": "No PDF uploaded"
            })

        # Create session if needed
        if not request.session.session_key:
            request.session.create()

        session_id = request.session.session_key

        # Upload directory
        upload_dir = os.path.join(
            settings.MEDIA_ROOT,
            "uploads"
        )

        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded PDF
        pdf_path = os.path.join(
            upload_dir,
            f"{session_id}.pdf"
        )

        with open(pdf_path, "wb+") as destination:

            for chunk_data in pdf_file.chunks():

                destination.write(chunk_data)

        # Reset globals to prevent using old index if this upload fails
        index = None
        chunks = None

        # Create/load vectorstore
        try:

            index, chunks = initialize_rag(
                pdf_path,
                session_id
            )

        except Exception as e:

            print("RAG Initialization Error:", e)

            return JsonResponse({
                "message": f"Error processing PDF: {str(e)}"
            })

        # Clear previous chat history
        request.session["chat_history"] = []

        return JsonResponse({
            "message": "PDF uploaded successfully"
        })

    return JsonResponse({
        "message": "Invalid request"
    })


def clear_chat(request):

    request.session["chat_history"] = []

    return JsonResponse({
        "status": "cleared"
    })
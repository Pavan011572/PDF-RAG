import os
import requests
import base64
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """You are a document question-answering assistant.

Answer the user's question using the provided context.

Rules:
1. Use the retrieved context as the primary source of information.
2. Do not invent facts.
3. If the answer cannot be found in the context, clearly say:
   "I could not find the answer in the uploaded document."
4. Give a concise and clear answer.
5. When possible, mention the relevant source page.

Context:
{context}

Question:
{question}

Answer:"""


class LLMError(Exception):
    """Exception raised for LLM API configuration or execution failures."""
    pass


class LLMManager:
    """
    Manages LLM API connections for Google Gemini, Groq (Instant Free), and OpenAI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "gemini",
        model_name: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if "groq" in self.provider:
            self.model_name = model_name or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
        elif "gemini" in self.provider:
            self.model_name = model_name or os.getenv("GEMINI_MODEL") or "gemini-1.5-flash"
        else:
            self.model_name = model_name or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"

    def format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context chunks into structured text for the prompt."""
        if not context_chunks:
            return "No relevant context found in documents."

        formatted_parts = []
        for idx, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "Unknown Document")
            page = chunk.get("page_number", "Unknown Page")
            content = chunk.get("content", "").strip()
            formatted_parts.append(
                f"[Chunk {idx}] (Source: {source}, Page: {page})\n{content}"
            )

        return "\n\n".join(formatted_parts)

    def render_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Render QA prompt combining template, context, and question."""
        context_str = self.format_context(context_chunks)
        return PROMPT_TEMPLATE.format(context=context_str, question=question)

    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using selected provider."""
        if not context_chunks:
            return "I could not find the answer in the uploaded document."

        if not self.api_key or self.api_key.strip() in ["your_api_key_here", "your_gemini_api_key_here", "your_groq_api_key_here"]:
            if "groq" in self.provider:
                raise LLMError(
                    "Groq API key is missing. Get your instant FREE key at "
                    "https://console.groq.com/keys and paste it in the sidebar."
                )
            elif "gemini" in self.provider:
                raise LLMError(
                    "Google Gemini API key is missing. Get your FREE key at "
                    "https://aistudio.google.com/app/apikey and paste it in the sidebar."
                )
            else:
                raise LLMError("OpenAI API key is missing. Please enter it in the sidebar.")

        prompt_text = self.render_prompt(question, context_chunks)

        if "groq" in self.provider:
            return self._generate_groq(prompt_text)
        elif "gemini" in self.provider:
            return self._generate_gemini(prompt_text)
        else:
            return self._generate_openai(prompt_text)

    def _generate_groq(self, prompt_text: str) -> str:
        """Generate answer via Groq Cloud API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a precise document question-answering assistant."},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.0
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                error_detail = response.json().get("error", {}).get("message", response.text)
                raise LLMError(f"Groq API Error ({response.status_code}): {error_detail}")

            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return "I could not find the answer in the uploaded document."

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Error communicating with Groq API: {str(e)}")

    def _generate_gemini(self, prompt_text: str) -> str:
        """Generate answer via Google Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0
            }
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                error_detail = response.json().get("error", {}).get("message", response.text)
                raise LLMError(f"Gemini API Error ({response.status_code}): {error_detail}")

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "I could not find the answer in the uploaded document."

            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
            return "I could not find the answer in the uploaded document."

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Error communicating with Gemini API: {str(e)}")

    def _generate_openai(self, prompt_text: str) -> str:
        """Generate answer via OpenAI API."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.0
            )

            response = llm.invoke([HumanMessage(content=prompt_text)])
            answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
            return answer

        except Exception as e:
            raise LLMError(f"OpenAI API Error: {str(e)}")

    def transcribe_image(self, image_path: str) -> str:
        """Transcribe text from a page image (handwritten notes or scanned PDF) using Vision AI."""
        if not os.path.exists(image_path):
            return ""

        if not self.api_key or self.api_key.strip() in ["your_api_key_here", "your_gemini_api_key_here", "your_groq_api_key_here"]:
            return ""

        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""

        prompt = "Transcribe all text from this page accurately, including handwritten notes, bullet points, headers, and diagrams. Return only the extracted text."

        if "groq" in self.provider:
            return self._transcribe_groq_vision(base64_image, prompt)
        elif "gemini" in self.provider:
            return self._transcribe_gemini_vision(base64_image, prompt)
        else:
            return self._transcribe_openai_vision(base64_image, prompt)

    def _transcribe_groq_vision(self, base64_image: str, prompt: str) -> str:
        """Transcribe image using Groq Vision model."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": os.getenv("GROQ_VISION_MODEL") or "qwen/qwen3.6-27b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            return ""
        except Exception as e:
            print(f"Groq Vision transcription warning: {e}")
            return ""

    def _transcribe_gemini_vision(self, base64_image: str, prompt: str) -> str:
        """Transcribe image using Google Gemini Vision model."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0}
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            return ""
        except Exception as e:
            print(f"Gemini Vision transcription warning: {e}")
            return ""

    def _transcribe_openai_vision(self, base64_image: str, prompt: str) -> str:
        """Transcribe image using OpenAI Vision model."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            return ""
        except Exception as e:
            print(f"OpenAI Vision transcription warning: {e}")
            return ""


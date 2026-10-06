"""
LLM client for FactCheckLIAR.

Provides integration with Ollama for LLM-based response generation,
with fallback to template-based responses when Ollama is unavailable.
"""

import os
from typing import Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default configuration from .env
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat")


def check_ollama_availability(api_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """
    Check if Ollama is available and responding.

    Args:
        api_url: Base URL of the Ollama API

    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get(f"{api_url}/api/tags", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def generate_ollama_response(
    prompt: str,
    api_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL
) -> Optional[str]:
    """
    Generate a response using Ollama's API.

    Args:
        prompt: The prompt to send to the LLM
        api_url: Base URL of the Ollama API
        model: Model name to use

    Returns:
        Generated response text, or None if generation failed
    """
    try:
        response = requests.post(
            f"{api_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        return None
    except (requests.ConnectionError, requests.Timeout, requests.JSONDecodeError):
        return None


def build_fact_check_prompt(
    query: str,
    retrieved_claim: dict,
    predicted_label: str,
    verbose: bool = False
) -> str:
    """
    Build a prompt for the LLM to generate a fact-check response.

    Args:
        query: The user's claim to fact-check
        retrieved_claim: Dictionary containing the retrieved similar claim data
        predicted_label: The predicted veracity label
        verbose: Whether to request a detailed response

    Returns:
        Formatted prompt string for the LLM
    """
    # Map labels to descriptions
    label_descriptions = {
        "pants-fire": "categorically false",
        "false": "false",
        "barely-true": "mostly false",
        "half-true": "partially true",
        "mostly-true": "mostly true",
        "true": "true"
    }
    label_desc = label_descriptions.get(predicted_label, predicted_label)

    # Format speaker name
    speaker = retrieved_claim.get('speaker', 'Unknown')
    speaker = speaker.replace('-', ' ').title()

    if verbose:
        prompt = f"""You are a fact-checking assistant. Analyze the following claim and provide a detailed fact-check response.

User's Claim: "{query}"

Similar Claim from Database:
- Statement: "{retrieved_claim.get('statement', '')}"
- Speaker: {speaker} ({retrieved_claim.get('job_title', 'N/A')})
- Context: {retrieved_claim.get('context', 'N/A')}
- Original Label: {retrieved_claim.get('label', 'N/A')}

Our AI classifier predicts this claim is: {label_desc}

Please provide a detailed fact-check response that:
1. Explains the relationship between the user's claim and the similar claim found
2. Discusses the evidence and context
3. Provides a clear verdict based on the predicted label

Keep your response informative but concise (2-3 paragraphs)."""
    else:
        prompt = f"""You are a fact-checking assistant. Provide a brief, clear verdict on this claim.

User's Claim: "{query}"

Similar claim found from {speaker}: "{retrieved_claim.get('statement', '')}"

Our AI classifier predicts this claim is: {label_desc}

Provide a single, clear sentence stating the verdict. Be direct and factual."""

    return prompt


def generate_template_response(
    query: str,
    retrieved_claim: dict,
    predicted_label: str,
    verbose: bool = False
) -> str:
    """
    Generate a template-based response (fallback when LLM unavailable).

    Args:
        query: The user's claim to fact-check
        retrieved_claim: Dictionary containing the retrieved similar claim data
        predicted_label: The predicted veracity label
        verbose: Whether to provide a detailed response

    Returns:
        Formatted response string
    """
    # Map labels to descriptions
    label_descriptions = {
        "pants-fire": "categorically false",
        "false": "false",
        "barely-true": "mostly false",
        "half-true": "partially true",
        "mostly-true": "mostly true",
        "true": "true"
    }
    label_desc = label_descriptions.get(predicted_label, predicted_label)

    # Format speaker name
    speaker = retrieved_claim.get('speaker', 'Unknown')
    speaker = speaker.replace('-', ' ').title()

    if not verbose:
        return (
            f"If you are referring to a claim by {speaker} that "
            f"{retrieved_claim.get('statement', '')}\n"
            f"It is {label_desc}."
        )
    else:
        return (
            f"Claim: \"{query}\"\n"
            f"Predicted Label: {predicted_label}\n\n"
            "Supporting Evidence from the Dataset:\n"
            f"- Statement: \"{retrieved_claim.get('statement', '')}\"\n"
            f"- Speaker: {retrieved_claim.get('speaker', '')} ({retrieved_claim.get('job_title', '')})\n"
            f"- Context: {retrieved_claim.get('context', '')}\n"
            f"- Dataset Label: {retrieved_claim.get('label', '')}\n\n"
            f"If you are referring to the claim above, it is {label_desc}."
        )


def generate_response(
    query: str,
    retrieved_claim: dict,
    predicted_label: str,
    verbose: bool = False,
    use_llm: bool = True
) -> str:
    """
    Generate a fact-check response, using LLM if available or template fallback.

    Args:
        query: The user's claim to fact-check
        retrieved_claim: Dictionary containing the retrieved similar claim data
        predicted_label: The predicted veracity label
        verbose: Whether to provide a detailed response
        use_llm: Whether to attempt LLM generation (False = use template only)

    Returns:
        Generated response string
    """
    if not use_llm:
        return generate_template_response(query, retrieved_claim, predicted_label, verbose)

    # Check if Ollama is available
    if not check_ollama_availability():
        print("Warning: Ollama is not available. Using template-based response.")
        return generate_template_response(query, retrieved_claim, predicted_label, verbose)

    # Build prompt and generate response
    prompt = build_fact_check_prompt(query, retrieved_claim, predicted_label, verbose)
    llm_response = generate_ollama_response(prompt)

    if llm_response:
        return llm_response
    else:
        print("Warning: LLM generation failed. Using template-based response.")
        return generate_template_response(query, retrieved_claim, predicted_label, verbose)

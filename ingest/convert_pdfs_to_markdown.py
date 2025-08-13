import os
from pathlib import Path
from dotenv import load_dotenv

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser


# -------------------
# Utility Functions
# -------------------
def load_gemini_api_key() -> str:
    """
    Load the GEMINI_API_KEY from .env file.
    Raises ValueError if key is missing.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    return api_key


def prepare_output_dir(path: Path) -> None:
    """
    Ensure that the output directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def create_pdf_converter(gemini_api_key: str) -> PdfConverter:
    """
    Initialize and return a PdfConverter instance with Marker configuration.
    """
    config = {
        "output_format": "markdown",      # output format: markdown, chunks, json
        "llm_model": "gemini-2.0-flash",  # Gemini model name
        "gemini_api_key": gemini_api_key,
        "use_llm": False,                  # enable image-to-text
        "use_ocr": False,                  # optional OCR
        "extract_images": False             # skip image extraction when not using LLM
    }

    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()  # not used without paid service
    )

    return converter


# -------------------
# Main Processing
# -------------------
def main():
    input_dir = Path("../docs")
    output_dir = Path("../docs/markdown/")
    prepare_output_dir(output_dir)

    gemini_api_key = load_gemini_api_key()
    converter = create_pdf_converter(gemini_api_key)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        rendered = converter(str(pdf_path))  # returns rendered markdown as string

        # Convert rendered object to string if necessary
        if hasattr(rendered, "as_string"):
            md_text = rendered.as_string()
        else:
            md_text = str(rendered)

        # Save markdown output
        output_file = output_dir / f"{pdf_path.stem}.md"
        output_file.write_text(md_text, encoding="utf-8")
        print(f"Saved to {output_file}")

    print("All PDFs processed successfully.")


# -------------------
# Entry Point
# -------------------
if __name__ == "__main__":
    main()

import os
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

@dataclass
class AppConfig:
    image_unavailable_message: str
    text_extensions: list[str]
    image_extensions: list[str]
    pdf_extensions: list[str]
    docx_extensions: list[str]
    inline_text_char_limit: int
    max_inline_messages: int
    upload_root_dir: Path
    mcp_tool_url:str

    use_embedding: bool
    use_file_read_write_tools: bool


@dataclass
class DBConfig:
    host: str
    port: int
    name: str
    user: str
    password: str

@dataclass
class EmbedderSettings:
    embed_model: str
    chunk_size: int
    chunk_overlap: int

env_path = Path(__file__).resolve().parent.parent /'.env'

try:
    load_dotenv(dotenv_path=env_path)
except FileNotFoundError:
    print("Env file not found!")

class EnvService:
    def get_image_fail_messages(self) -> str | None:
        return os.getenv("IMAGE_FAIL_MESSAGES", "")

    def get_app_config(self) -> AppConfig:
        return AppConfig(
            image_unavailable_message=os.getenv("IMAGE_UNAVAILABLE_MESSAGE", ""),
            text_extensions=literal_eval(os.getenv("TEXT_EXTENSIONS", "[]")),
            image_extensions=literal_eval(os.getenv("IMAGE_EXTENSIONS", "[]")),
            pdf_extensions=literal_eval(os.getenv("PDF_EXTENSIONS", "[]")),
            docx_extensions=literal_eval(os.getenv("DOCX_EXTENSIONS", "[]")),
            inline_text_char_limit=int(os.getenv("INLINE_TEXT_CHAR_LIMIT", "0")),
            max_inline_messages=int(os.getenv("MAX_INLINE_MESSAGES", "0")),
            upload_root_dir= Path("uploads"),
            mcp_tool_url = os.getenv("MCP_TOOL_URL", ""),
            use_embedding=bool(os.getenv("USE_EMBEDDING", "0")),
            use_file_read_write_tools=bool(os.getenv("USE_FILE_READWRITE", "0")),
        )

    def get_db_config(self) -> DBConfig:
        return DBConfig(
            host=os.getenv("DB_HOST", ""),
            port=int(os.getenv("DB_PORT") or 0),
            name=os.getenv("DB_NAME", ""),
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", ""),
        )

    def get_ollama_api_key(self) -> str:
        return os.getenv("OLLAMA_API_KEY", "")

    def get_embedder_settings(self):
        return EmbedderSettings(
            embed_model=os.getenv("EMBED_MODEL", ""),
            chunk_size=int(os.getenv("CHUNK_SIZE", "0")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "0")),
        )
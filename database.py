import os
from pathlib import Path
from typing import List, Optional

# Здесь я собираю всю работу с базой знаний:
# - загрузка PDF из директории с книгами;
# - нарезка на чанки фиксированного размера;
# - построение/подключение ChromaDB.
# Это позволяет держать индексацию отдельно от Telegram-логики и легко
# переиспользовать в любых других сервисах.

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


def get_knowledge_dir() -> Path:
    """
    Возвращаем путь к директории с PDF-книгами.
    По умолчанию подстраиваюсь под постановку задачи — папка /books
    (относительно корня проекта). При этом оставляю гибкость через ENV.
    """
    dir_path = os.getenv("KNOWLEDGE_DIR", "./books")
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_chroma_persist_dir() -> str:
    """
    Путь к директории, где Chroma будет хранить локальную векторную базу.
    Держу его в ENV, чтобы можно было разносить base на разные диски/volume.
    """
    return os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def load_pdf_documents() -> List:
    """
    Загружаем все PDF-документы из директории с книгами.
    Каждый PDF превращаем в список документов LangChain.
    """
    knowledge_dir = get_knowledge_dir()
    pdf_files = sorted(knowledge_dir.glob("*.pdf"))

    if not pdf_files:
        # Предпочитаю явно подсветить отсутствие данных, чтобы не удивляться
        # пустым ответам RAG на проде.
        print(f"[database] В директории {knowledge_dir} нет PDF-файлов.")
        return []

    all_docs: List = []
    for pdf_path in pdf_files:
        print(f"[database] Загружаю PDF: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)

    print(f"[database] Всего загружено сырых документов: {len(all_docs)}")
    return all_docs


def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Режем документы на чанки.

    В этой воронке я сознательно выбираю размер чанка ~1000 символов
    (как ты и просил), с небольшим overlap. Этого достаточно, чтобы:
    - не терять маркетинговый контекст (пример, идея, кейс);
    - не раздувать запросы к LLM.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"[database] После сплита документов: {len(split_docs)} чанков")
    return split_docs


def get_document_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Эмбеддинги для ДОКУМЕНТОВ.

    Здесь я жёстко задаю:
    - model="models/text-embedding-004" — полный путь, как требует Gemini в 2026;
    - task_type="retrieval_document" — говорим, что этот вектор будет храниться
      в базе и использоваться как "эталон" при поиске.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document",
    )


def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Эмбеддинги для ЗАПРОСОВ (вопросов пользователя).

    Здесь важно:
    - та же модель "models/text-embedding-004";
    - task_type="retrieval_query" — это вектор для поиска по базе документов.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_query",
    )


def build_knowledge_base(force_rebuild: bool = False) -> Optional[Chroma]:
    """
    Полная индексация всех PDF в ChromaDB.

    - Если база уже существует и force_rebuild=False — просто подключаемся.
    - Если force_rebuild=True — пересобираем базу с нуля.

    Я делаю это синхронным "офлайновым" шагом:
    его удобно вызывать отдельной командой:
        python database.py
    а не тащить индексацию при каждом старте бота.
    """
    persist_dir = get_chroma_persist_dir()

    # Если база уже есть и пересборка не требуется — просто подключаем её.
    if not force_rebuild and os.path.isdir(persist_dir) and os.listdir(persist_dir):
        print(f"[database] Используем уже существующую ChromaDB в {persist_dir}")
        vectordb = Chroma(
            persist_directory=persist_dir,
            # Для уже готовой базы при обычном подключении нам нужны только query-эмбеддинги.
            embedding_function=get_query_embeddings(),
        )
        return vectordb

    print("[database] Запускаю полную пересборку базы знаний...")
    documents = load_pdf_documents()
    if not documents:
        print("[database] Нет документов для индексации. База не построена.")
        return None

    split_docs = split_documents(documents)

    # На этапе построения мы явно говорим: "мы индексируем документы"
    # через task_type="retrieval_document".
    doc_embeddings = get_document_embeddings()

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=doc_embeddings,
        persist_directory=persist_dir,
    )

    print(f"[database] База знаний успешно проиндексирована и сохранена в {persist_dir}")
    return vectordb


def get_vectorstore() -> Optional[Chroma]:
    """
    Возвращаем подключённый Chroma vectorstore.

    Если база ещё не построена — возвращаем None, и агент
    будет работать только на "мозгах" LLM без RAG.
    """
    persist_dir = get_chroma_persist_dir()
    if not os.path.isdir(persist_dir) or not os.listdir(persist_dir):
        print(f"[database] ChromaDB в {persist_dir} пока нет. Запустите индексацию: `python database.py`.")
        return None

    vectordb = Chroma(
        persist_directory=persist_dir,
        # Для поиска достаточно query-эмбеддингов (retrieval_query).
        embedding_function=get_query_embeddings(),
    )
    print(f"[database] Подключена существующая ChromaDB из {persist_dir}")
    return vectordb


if __name__ == "__main__":
    """
    Небольшой CLI-хук, чтобы руками перегружать базу:
    - python database.py                    # обычная индексация
    - FORCE_REBUILD=true python database.py # принудительная пересборка
    """
    force = os.getenv("FORCE_REBUILD", "false").lower() in ("1", "true", "yes")
    build_knowledge_base(force_rebuild=force)


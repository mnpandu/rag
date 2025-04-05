from langchain.text_splitter import CharacterTextSplitter

def load_and_split_text(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    return [t for t in splitter.split_text(text) if t.strip()]
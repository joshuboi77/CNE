from sqlalchemy.orm import Session
from database import Author, Work, Word, create_tables
from word_parser import WordParser
import os
from pathlib import Path

class DataLoader:
    def __init__(self, db: Session):
        self.db = db
        self.parser = WordParser()
    
    def load_author(self, author_name: str) -> Author:
        """
        Get or create an author in the database.
        """
        author = self.db.query(Author).filter(Author.name == author_name).first()
        if not author:
            author = Author(name=author_name)
            self.db.add(author)
            self.db.commit()
            self.db.refresh(author)
        return author
    
    def load_work(self, author: Author, metadata: dict, file_path: str) -> Work:
        """
        Get or create a work in the database.
        """
        work = self.db.query(Work).filter(Work.file_path == file_path).first()
        if not work:
            work = Work(
                title=metadata.get('title', 'Unknown Title'),
                author_id=author.id,
                year=metadata.get('year'),
                language=metadata.get('languages', ['en'])[0] if metadata.get('languages') else 'en',
                source=metadata.get('source'),
                gutendex_id=metadata.get('gutendex_id'),
                file_path=file_path
            )
            self.db.add(work)
            self.db.commit()
            self.db.refresh(work)
        return work
    
    def load_words(self, work: Work, words_data: list):
        """
        Load words for a work into the database.
        """
        # Clear existing words for this work
        self.db.query(Word).filter(Word.work_id == work.id).delete()
        
        # Insert new words
        word_objects = []
        for original_word, cleaned_word, position in words_data:
            word_obj = Word(
                word=original_word,
                cleaned_word=cleaned_word,
                work_id=work.id,
                position=position
            )
            word_objects.append(word_obj)
        
        # Bulk insert for better performance
        self.db.bulk_save_objects(word_objects)
        self.db.commit()
        
        print(f"Loaded {len(word_objects)} words for work: {work.title}")
    
    def load_corpus(self, corpus_dir: str):
        """
        Load the entire corpus into the database.
        """
        print("Starting corpus loading...")
        
        # Process all files
        processed_data = self.parser.process_corpus(corpus_dir)
        
        total_works = 0
        total_words = 0
        
        for data in processed_data:
            try:
                # Load author
                author = self.load_author(data['author'])
                
                # Load work
                work = self.load_work(author, data['metadata'], data['text_file_path'])
                
                # Load words
                self.load_words(work, data['words'])
                
                total_works += 1
                total_words += len(data['words'])
                
            except Exception as e:
                print(f"Error processing {data.get('text_file_path', 'unknown')}: {e}")
                continue
        
        print(f"Corpus loading completed!")
        print(f"Total works loaded: {total_works}")
        print(f"Total words loaded: {total_words}")

def initialize_database(corpus_dir: str):
    """
    Initialize the database and load all data.
    """
    # Create tables
    create_tables()
    
    # Create database session
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        # Load data
        loader = DataLoader(db)
        loader.load_corpus(corpus_dir)
    finally:
        db.close()

if __name__ == "__main__":
    # Initialize database with the corpus
    corpus_path = "/Volumes/Storage/CNE/data/cne_corpus"
    initialize_database(corpus_path)


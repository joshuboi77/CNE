from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database setup
DATABASE_URL = "sqlite:///./cne_words.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Author(Base):
    __tablename__ = "authors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to works
    works = relationship("Work", back_populates="author")

class Work(Base):
    __tablename__ = "works"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False, index=True)
    author_id = Column(Integer, ForeignKey("authors.id"), nullable=False)
    year = Column(Integer, index=True)
    language = Column(String(10), default="en")
    source = Column(String(255))
    gutendex_id = Column(Integer, unique=True)
    file_path = Column(String(1000), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    author = relationship("Author", back_populates="works")
    words = relationship("Word", back_populates="work")

class Word(Base):
    __tablename__ = "words"
    
    id = Column(Integer, primary_key=True, index=True)
    word = Column(String(255), nullable=False, index=True)
    cleaned_word = Column(String(255), nullable=False, index=True)
    work_id = Column(Integer, ForeignKey("works.id"), nullable=False)
    position = Column(Integer, nullable=False)  # Position in the text
    chapter = Column(String(100))  # Chapter or section if available
    line_number = Column(Integer)  # Line number in the original text
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    work = relationship("Work", back_populates="words")
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_word_work', 'word', 'work_id'),
        Index('idx_cleaned_word_work', 'cleaned_word', 'work_id'),
        Index('idx_position_work', 'position', 'work_id'),
    )

# Create all tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


import re
import json
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class WordParser:
    def __init__(self):
        # Common words to filter out (can be expanded)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'she',
            'him', 'her', 'his', 'their', 'this', 'these', 'those', 'or',
            'but', 'if', 'when', 'where', 'why', 'how', 'what', 'who',
            'which', 'there', 'here', 'then', 'than', 'so', 'up', 'down',
            'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'very', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall', 'do',
            'does', 'did', 'have', 'had', 'having', 'been', 'being'
        }
    
    def clean_word(self, word: str) -> str:
        """
        Clean a word by removing punctuation, converting to lowercase,
        and handling special characters.
        """
        # Remove all non-alphabetic characters except hyphens in the middle
        cleaned = re.sub(r'[^a-zA-Z\-]', '', word)
        
        # Convert to lowercase
        cleaned = cleaned.lower()
        
        # Remove leading/trailing hyphens
        cleaned = cleaned.strip('-')
        
        # Remove empty strings or single characters (except 'a' and 'i')
        if len(cleaned) <= 1 and cleaned not in ['a', 'i']:
            return ""
        
        return cleaned
    
    def is_valid_word(self, word: str) -> bool:
        """
        Check if a word is valid for storage.
        """
        cleaned = self.clean_word(word)
        
        # Must have at least 2 characters or be 'a' or 'i'
        if len(cleaned) < 2 and cleaned not in ['a', 'i']:
            return False
        
        # Skip words that are only punctuation
        if not re.search(r'[a-zA-Z]', cleaned):
            return False
        
        return True
    
    def extract_words_from_text(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Extract words from text and return list of (original_word, cleaned_word, position).
        """
        words = []
        # Split on whitespace and common punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        for i, token in enumerate(tokens):
            cleaned = self.clean_word(token)
            if self.is_valid_word(token):
                words.append((token, cleaned, i))
        
        return words
    
    def parse_metadata(self, meta_file_path: str) -> Dict:
        """
        Parse metadata from JSON file.
        """
        try:
            with open(meta_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error parsing metadata {meta_file_path}: {e}")
            return {}
    
    def process_text_file(self, text_file_path: str, meta_file_path: str) -> Dict:
        """
        Process a single text file and return structured data.
        """
        # Read text file
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading text file {text_file_path}: {e}")
            return {}
        
        # Parse metadata
        metadata = self.parse_metadata(meta_file_path)
        
        # Extract words
        words = self.extract_words_from_text(text)
        
        return {
            'metadata': metadata,
            'words': words,
            'text_file_path': text_file_path,
            'meta_file_path': meta_file_path
        }
    
    def process_corpus(self, corpus_dir: str) -> List[Dict]:
        """
        Process all text files in the corpus directory.
        """
        corpus_path = Path(corpus_dir)
        results = []
        
        # Find all author directories
        for author_dir in corpus_path.iterdir():
            if author_dir.is_dir():
                print(f"Processing author: {author_dir.name}")
                
                # Find all text files in author directory
                for text_file in author_dir.glob("*.txt"):
                    meta_file = text_file.with_suffix('.meta.json')
                    
                    if meta_file.exists():
                        print(f"  Processing: {text_file.name}")
                        result = self.process_text_file(str(text_file), str(meta_file))
                        if result:
                            result['author'] = author_dir.name
                            results.append(result)
                    else:
                        print(f"  Warning: No metadata file found for {text_file.name}")
        
        return results

# Example usage
if __name__ == "__main__":
    parser = WordParser()
    
    # Test with a sample text
    sample_text = "The quick brown fox jumps over the lazy dog. It's a beautiful day!"
    words = parser.extract_words_from_text(sample_text)
    
    print("Sample word extraction:")
    for original, cleaned, pos in words:
        print(f"  {pos}: '{original}' -> '{cleaned}'")


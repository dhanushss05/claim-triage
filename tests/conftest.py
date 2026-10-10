import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("PROMPTS_DIR", os.path.join(ROOT, "prompts"))

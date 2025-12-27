import os

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print("Using model:", OPENAI_MODEL)


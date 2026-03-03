"""
Maps course_id -> display name for the course recommendation UI.
Loads from pinecone_database_stuff/combined_course_data.json if present; otherwise empty dict.
"""
import os
import json

course_names = {}
_paths = [
    "pinecone_database_stuff/combined_course_data.json",
    os.path.join(os.path.dirname(__file__), "pinecone_database_stuff", "combined_course_data.json"),
]
for _path in _paths:
    if os.path.isfile(_path):
        try:
            with open(_path, "r") as f:
                data = json.load(f)
            course_names = {item["course_id"]: item["course_id"] for item in data if "course_id" in item}
        except Exception:
            pass
        break

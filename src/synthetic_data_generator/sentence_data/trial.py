import json
import pandas as pd


def merge_json_to_csv(course_path, cv_path, job_path, output_csv_path):
    """
    Merges JSON files (course, cv, job) into a single CSV.
    Each JSON file is assumed to contain an array of objects, where each object has:
        - escoid, preferredLabel, description, and sentences (with explicit and implicit lists).
    For each sentence (explicit and implicit), a row is created in the output CSV.

    The resulting CSV has columns:
        - escoid, preferredLabel, description, sentence, sentence_type, extract

    Parameters:
    - course_path: str, path to the course JSON file.
    - cv_path: str, path to the cv JSON file.
    - job_path: str, path to the job JSON file.
    - output_csv_path: str, path where the merged CSV will be saved.

    Returns:
    - A pandas DataFrame of the merged data.
    """
    rows = []

    def process_file(file_path, source_label):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            escoid = item.get("escoid")
            preferredLabel = item.get("preferredLabel")
            description = item.get("description")
            sentences = item.get("sentences", {})
            # Process explicit sentences
            for sentence in sentences.get("explicit", []):
                rows.append(
                    {
                        "escoid": escoid,
                        "preferredLabel": preferredLabel,
                        "description": description,
                        "sentence": sentence,
                        "sentence_type": "explicit",
                        "extract": source_label,
                        "sentence_type": "explicit",
                    }
                )
            # Process implicit sentences
            for sentence in sentences.get("implicit", []):
                rows.append(
                    {
                        "escoid": escoid,
                        "preferredLabel": preferredLabel,
                        "description": description,
                        "sentence": sentence,
                        "sentence_type": "implicit",
                        "extract": source_label,
                        "sentence_type": "implicit",
                    }
                )

    process_file(course_path, "course")
    process_file(cv_path, "cv")
    process_file(job_path, "job")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    return df


# Example usage:
df_merged = merge_json_to_csv(
    "./data/course.json",
    "./data/cv.json",
    "./data/job.json",
    "./data/merged_dataset.csv",
)


print(df_merged.info())
print(df_merged.describe())
print(df_merged.head())

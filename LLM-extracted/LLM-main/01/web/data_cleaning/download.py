"""
data_cleaning/download.py
=========================
Downloads FAQ markdown files from the DataTalksClub/faq GitHub repo.

Downloads the main branch zip, extracts only the _questions directory
for all four courses into data/raw/.

Input:  None (downloads from GitHub)
Output: data_cleaning/data/raw/<course>/<section>/*.md

Run:    uv run python data_cleaning/download.py
"""
import os
import urllib.request
import zipfile
import shutil

RAW_DIR = 'data_cleaning/data/raw'
REPO_URL = 'https://github.com/DataTalksClub/faq/archive/refs/heads/main.zip'
COURSES = [
    'llm-zoomcamp',
    'mlops-zoomcamp',
    'data-engineering-zoomcamp',
    'machine-learning-zoomcamp',
]


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, 'faq-main.zip')

    # Download
    print(f'Downloading {REPO_URL} ...')
    urllib.request.urlretrieve(REPO_URL, zip_path)
    print(f'Downloaded → {zip_path}')

    # Extract each course
    for course in COURSES:
        course_dir = os.path.join(RAW_DIR, course)
        os.makedirs(course_dir, exist_ok=True)

        source_prefix = f'faq-main/_questions/{course}/'
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all files for this course
            course_files = [
                name for name in zf.namelist()
                if name.startswith(source_prefix) and name.endswith('.md')
            ]

            if not course_files:
                print(f'  No files found for {course}')
                continue

            for name in course_files:
                # Extract to temp location
                zf.extract(name, RAW_DIR)
                
                # Move to course directory, flattening structure
                source_path = os.path.join(RAW_DIR, name)
                dest_name = name[len(source_prefix):]  # Remove prefix
                dest_path = os.path.join(course_dir, dest_name)
                
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(source_path, dest_path)

            # Clean up empty faq-main directory
            faq_main_dir = os.path.join(RAW_DIR, 'faq-main')
            if os.path.exists(faq_main_dir):
                shutil.rmtree(faq_main_dir)

            file_count = sum(1 for _, _, files in os.walk(course_dir) for f in files if f.endswith('.md'))
            print(f'  {course}: {file_count} files')

    # Clean up zip
    os.remove(zip_path)
    print(f'\nDone. Files extracted to {RAW_DIR}/')


if __name__ == '__main__':
    main()

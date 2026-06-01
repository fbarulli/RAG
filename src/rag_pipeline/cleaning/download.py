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
from rag_pipeline.cleaning.core.paths import Paths

RAW_DIR = str(Paths.raw_dir())


def download_repo(url, path):
    """Downloads a zip file from a given URL and saves it to a specified path."""
    try:
        print(f'Downloading {url} ...')
        urllib.request.urlretrieve(url, path)
        print(f'Downloaded → {path}')
    except Exception as e:
        print(f'Download failed: {e}')
        return False
    return True


def extract_course_files(zip_file, course, source_prefix, dest_dir):
    """Extracts markdown files for a given course from a zip file and saves them to a specified directory."""
    try:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            course_files = [
                name for name in zf.namelist()
                if name.startswith(source_prefix) and name.endswith('.md')
            ]
            if not course_files:
                print(f' No files found for {course}')
                return
            for name in course_files:
                zf.extract(name, RAW_DIR)
                source_path = os.path.join(RAW_DIR, name)
                dest_name = name[len(source_prefix):]
                dest_path = os.path.join(dest_dir, dest_name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(source_path, dest_path)
            file_count = sum(
                1 for _, _, fnames in os.walk(dest_dir)
                for f in fnames if f.endswith('.md')
            )
            print(f' {course}: {file_count} files')
    except Exception as e:
        print(f'Extraction failed for {course}: {e}')


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, 'faq-main.zip')

    # Pull configuration from defaults.json
    repo_url = Paths.data_download_url()
    courses = Paths.download_courses()

    if not download_repo(repo_url, zip_path):
        return

    for course in courses:
        course_dir = os.path.join(RAW_DIR, course)
        os.makedirs(course_dir, exist_ok=True)
        source_prefix = f'faq-main/_questions/{course}/'
        extract_course_files(zip_path, course, source_prefix, course_dir)

    faq_main_dir = os.path.join(RAW_DIR, 'faq-main')
    if os.path.exists(faq_main_dir):
        shutil.rmtree(faq_main_dir)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f'\nDone. Files extracted to {RAW_DIR}/')


if __name__ == '__main__':
    main()
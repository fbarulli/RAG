"""
data_cleaning/extract_metadata.py
==================================
Extracts image references from the raw markdown files and saves them
to a separate metadata file for future use.

Input:  data_cleaning/data/raw/<course>/<section>/*.md
Output: data_cleaning/metadata/images.json
        {image_id: {path, description, course, section, source_file}}

Run:    uv run python data_cleaning/extract_metadata.py
"""
import os
import json
import yaml

RAW_DIR = 'data_cleaning/data/raw'
OUTPUT = 'data_cleaning/metadata/images.json'


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    
    all_images = {}
    
    for course in sorted(os.listdir(RAW_DIR)):
        course_path = os.path.join(RAW_DIR, course)
        if not os.path.isdir(course_path):
            continue

        for section in sorted(os.listdir(course_path)):
            section_path = os.path.join(course_path, section)
            if not os.path.isdir(section_path):
                continue

            for filename in sorted(os.listdir(section_path)):
                if not filename.endswith('.md'):
                    continue

                filepath = os.path.join(section_path, filename)
                
                with open(filepath) as f:
                    content = f.read()
                
                if not content.startswith('---'):
                    continue
                
                end = content.find('---', 3)
                if end == -1:
                    continue
                
                try:
                    frontmatter = yaml.safe_load(content[3:end].strip()) or {}
                except yaml.YAMLError:
                    continue
                
                images = frontmatter.get('images', [])
                if isinstance(images, list):
                    for img in images:
                        if isinstance(img, dict) and 'id' in img:
                            img_id = img['id']
                            all_images[img_id] = {
                                'path': img.get('path', ''),
                                'description': img.get('description', ''),
                                'course': course,
                                'section': section,
                                'source_file': filename,
                            }

    with open(OUTPUT, 'w') as f:
        json.dump(all_images, f, indent=2)
    
    print(f'Extracted {len(all_images)} image references → {OUTPUT}')


if __name__ == '__main__':
    try:
        import yaml
    except ImportError:
        import subprocess
        subprocess.check_call(['uv', 'pip', 'install', 'pyyaml'])
        import yaml
    
    main()

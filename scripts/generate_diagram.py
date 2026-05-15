"""
scripts/generate_diagram.py
Automatically generates a Mermaid.js flowchart of the pipeline.
Usage: uv run python scripts/generate_diagram.py
"""
import os
import re
from pathlib import Path

def generate():
    base_dir = Path('.')
    pipeline_dir = base_dir / 'production_pipeline'
    src_dir = base_dir / 'src/rag_pipeline'
    
    # 1. Discover Pipeline Stages
    stages = []
    if pipeline_dir.exists():
        for d in sorted(pipeline_dir.iterdir()):
            if d.is_dir() and re.match(r"p\d+_", d.name):
                scripts = sorted([f.stem for f in d.glob("p*.py")])
                if scripts:
                    stages.append({"dir": d.name, "scripts": scripts})
    
    # 2. Discover Shared Libs
    libs = []
    if src_dir.exists():
        libs = sorted([f.stem for f in src_dir.glob("*.py") if f.stem != "__init__"])
    
    # 3. Discover Tests
    tests = []
    tests_dir = base_dir / 'tests'
    if tests_dir.exists():
        tests = sorted([f.stem for f in tests_dir.glob("*.py") if f.stem != "__init__"])

    # 4. Build Mermaid
    lines = ["flowchart TD"]
    
    # Shared Lib Subgraph
    lines.append("    subgraph Core[src/rag_pipeline]")
    for lib in libs:
        lines.append(f"    {lib}[{lib}.py]")
    lines.append("    end")
    
    # Pipeline Stages Subgraph
    lines.append("    subgraph Stages[production_pipeline/]")
    lines.append("        direction LR")
    prev_stage_id = None
    
    for stage in stages:
        stage_id = stage['dir']
        lines.append(f"        subgraph {stage_id}[{stage_id}]")
        for i, script in enumerate(stage['scripts']):
            script_id = f"{stage_id}_{script}"
            lines.append(f"        {script_id}[{script}.py]")
            
            # Link scripts within a stage sequentially
            if i > 0:
                prev_script_id = f"{stage_id}_{stage['scripts'][i-1]}"
                lines.append(f"        {prev_script_id} --> {script_id}")
        lines.append("        end")
        
        # Link to data outputs (heuristic based on common names)
        if "parse" in stage_id: lines.append(f"        {stage_id}_{stage['scripts'][-1]} -->|Write| Parsed[(parsed.jsonl)]")
        if "dedup" in stage_id: lines.append(f"        Parsed -->|Read| {stage_id}_{stage['scripts'][0]}")
        if "dedup" in stage_id: lines.append(f"        {stage_id}_{stage['scripts'][-1]} -->|Write| Clean[(clean.jsonl)]")
        if "eda" in stage_id: lines.append(f"        Clean -->|Read| {stage_id}_{stage['scripts'][0]}")
        if "eda" in stage_id: lines.append(f"        {stage_id}_{stage['scripts'][-1]} -->|Write| Summary[(eda_summary.json)]")
        if "download" in stage_id: lines.append(f"        {stage_id}_{stage['scripts'][-1]} -->|Write| Raw[(data/raw/)]")
        if "parse" in stage_id: lines.append(f"        Raw -->|Read| {stage_id}_{stage['scripts'][0]}")

    lines.append("    end")

    # Tests Subgraph
    if tests:
        lines.append("    subgraph Tests[tests/]")
        for t in tests:
            lines.append(f"    {t}[{t}.py]")
        lines.append("    end")
        
        # Link tests to pipeline
        lines.append("    p01_data_cleaning_p02_parse -.->|Verify| test_cleaning")
        lines.append("    parsed -.->|Verify| inspect_comparisons")

    # Output
    output = "\n".join(lines)
    output_file = Path('pipeline_diagram.md')
    with open(output_file, 'w') as f:
        f.write("```mermaid\n" + output + "\n```")
    
    print(f"✅ Generated {output_file} with {len(stages)} stages.")

if __name__ == '__main__':
    generate()

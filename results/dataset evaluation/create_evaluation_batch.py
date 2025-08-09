#!/usr/bin/env python3
"""
Script to create a JSONL file for OpenAI Batch API story evaluation.
Takes an array of paths to .txt files containing stories (one per line)
and creates a batch evaluation request file.
"""

import json
import os
import argparse
from pathlib import Path


def extract_model_name(file_path):
    """Extract model name from file path for custom_id prefix"""
    path = Path(file_path)
    # Extract the directory name which contains the model identifier
    parent_dir = path.parent.name
    return parent_dir


def create_evaluation_request(story, custom_id):
    """Create a single evaluation request in the batch format"""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "Evaluate the following story based on four criteria by assigning each a score from 0 to 100: 1. Originality: Rate the creativity and uniqueness of the story. 2. Coherence: Rate the logical flow and consistency of the story. 3. Grammar: Rate the grammatical correctness of the story. Ignore spacing and capitalization. 4. Quality: Rate the overall quality of the story. You should also provide a short explanation for your judgment."
                },
                {
                    "role": "user",
                    "content": f"Story to evaluate: {story}"
                }
            ],
            "max_tokens": 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "story_evaluation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "originality": {"type": "integer", "minimum": 0, "maximum": 100},
                            "coherence": {"type": "integer", "minimum": 0, "maximum": 100},
                            "grammar": {"type": "integer", "minimum": 0, "maximum": 100},
                            "quality": {"type": "integer", "minimum": 0, "maximum": 100}
                        },
                        "required": ["explanation", "originality", "coherence", "grammar", "quality"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        }
    }


def process_story_files(file_paths, output_file):
    """Process all story files and create batch evaluation JSONL"""
    requests = []
    story_counter = 200  # Start from 200 as requested
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        model_name = extract_model_name(file_path)
        print(f"Processing {file_path} (model: {model_name})")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                story = line.strip()
                if story:  # Skip empty lines
                    custom_id = f"{model_name}-{story_counter}"
                    request = create_evaluation_request(story, custom_id)
                    requests.append(request)
                    story_counter += 1
        
        print(f"  Added {line_num} stories from {file_path}")
    
    # Write all requests to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"\n✓ Created batch evaluation file: {output_file}")
    print(f"  Total requests: {len(requests)}")
    print(f"  Story IDs: 200-{story_counter-1}")
    
    return len(requests)


def main():
    parser = argparse.ArgumentParser(description='Create JSONL batch file for story evaluation')
    parser.add_argument('files', nargs='+', help='Paths to story text files')
    parser.add_argument('--output', '-o', default='out/dataset_tests/story_evaluation_batch.jsonl', 
                       help='Output JSONL file path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    total_requests = process_story_files(args.files, args.output)
    
    print(f"\nBatch file ready for OpenAI Batch API!")
    print(f"Upload {args.output} to create your batch job.")


if __name__ == '__main__':
    main()




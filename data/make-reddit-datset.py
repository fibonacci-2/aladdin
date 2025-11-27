import json
import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    """Remove URLs from text"""
    if not text:
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text.strip()

def process_reddit_data(folder_path):
    """Process Reddit JSONL files and output cleaned CSV"""
    folder = Path(folder_path)
    all_data = []
    subreddit_stats = {}
    
    jsonl_files = list(folder.glob("*.jsonl"))
    
    for file_path in jsonl_files:
        subreddit_name = file_path.stem.replace('_comments', '').replace('_posts', '')
        is_comment = 'comments' in file_path.stem
        
        if subreddit_name not in subreddit_stats:
            subreddit_stats[subreddit_name] = {'raw': 0, 'cleaned': 0}
        
        print(f"Processing {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    subreddit_stats[subreddit_name]['raw'] += 1
                    
                    if is_comment:
                        text_content = data.get('body', '')
                    else:
                        title = data.get('title', '')
                        body = data.get('selftext', '')
                        text_content = f"{title} {body}".strip()
                    
                    cleaned_text = clean_text(text_content)
                    
                    post_data = {
                        'selftext': cleaned_text,
                        'created_utc': data.get('created_utc'),
                        'id': data.get('id'),
                        'author': data.get('author'),
                        'subreddit': subreddit_name,
                        'is_comment': is_comment
                    }
                    
                    if (cleaned_text and 
                        len(cleaned_text.split()) >= 5):
                        
                        all_data.append(post_data)
                        subreddit_stats[subreddit_name]['cleaned'] += 1
                        
                except json.JSONDecodeError:
                    continue
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = folder / "cleaned_reddit_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
        
        print("\n=== PROCESSING STATISTICS ===")
        for subreddit, stats in subreddit_stats.items():
            print(f"{subreddit}: {stats['raw']} raw, {stats['cleaned']} cleaned")
        
        print(f"\nTotal posts/comments processed: {len(all_data)}")
    else:
        print("No valid data found to process.")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing JSONL files: ")
    process_reddit_data(folder_path)
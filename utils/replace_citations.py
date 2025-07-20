import sys
import pandas as pd
import re

def replace_citations(csv_path, tex_path):
    # Load mapping
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df['OldKey'], df['NewKey']))

    # Read LaTeX file
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace each old citation key with the new one
    for old_key, new_key in mapping.items():
        if old_key != '[NOT FOUND]':
            content = re.sub(r'\\cite\{\s*' + re.escape(old_key) + r'\s*\}', f'\\cite{{{new_key}}}', content)
            content = re.sub(r'\\citep\{\s*' + re.escape(old_key) + r'\s*\}', f'\\citep{{{new_key}}}', content)
            content = re.sub(r'\\citet\{\s*' + re.escape(old_key) + r'\s*\}', f'\\citet{{{new_key}}}', content)
            content = re.sub(r'\b' + re.escape(old_key) + r'\b', new_key, content)

    # Save the updated file
    output_path = tex_path.replace('.tex', '_updated.tex')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Updated file saved to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python replace_citations.py <mapping.csv> <file.tex>")
    else:
        replace_citations(sys.argv[1], sys.argv[2])

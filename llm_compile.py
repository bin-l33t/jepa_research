import sys
import os
import subprocess
import re

def extract_python(text):
    """
    Extracts the first valid block of Python code from the LLM response.
    """
    # Pattern 1: Markdown code blocks
    # We use [`]{3} to match triple backticks without breaking the UI parser
    pattern = r"[`]{3}(?:python)?(.*?)[`]{3}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Pattern 2: Raw code (fallback)
    code_start_pattern = r"^(import |from |def |class )"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(code_start_pattern, line.strip()):
            return '\n'.join(lines[i:]).strip()
            
    return None

def compile_code(output_file, prompt_files):
    """
    Reads prompt files, concatenates them, sends to Gemini, and writes output.
    """
    print(f"⚙️  Compiling target: {output_file}...")
    
    # 1. Read and combine all prompt/spec files
    full_prompt_text = ""
    for p_file in prompt_files:
        try:
            with open(p_file, 'r') as f:
                full_prompt_text += f.read() + "\n\n"
        except FileNotFoundError:
            print(f"❌ Error: Prompt file not found: {p_file}")
            sys.exit(1)

    sys_prompt = "You are a Python Code Generator. Output ONLY valid Python code. Do not use Markdown blocks. Do not explain. Do not launch interactive sessions."
    final_prompt = f"{sys_prompt}\n\n{full_prompt_text}"

    # 2. Call Gemini CLI safely (shell=False avoids quote breaking)
    # We pass the prompt as a direct argument list item, not a shell string
    cmd = ['gemini', '--model', 'gemini-3-pro-preview', final_prompt]
    
    try:
        # shell=False is crucial here to prevent shell expansion of quotes
        result = subprocess.run(cmd, capture_output=True, text=True, shell=False)
    except Exception as e:
        print(f"❌ Subprocess execution failed: {e}")
        sys.exit(1)
    
    if result.returncode != 0:
        print(f"❌ API Failure: {result.stderr}")
        sys.exit(1)
        
    code = extract_python(result.stdout)

    if not code:
        print("❌ Error: No valid Python code found in response.")
        # Debug: Print a snippet of what we got
        print(f"DEBUG Response snippet: {result.stdout[:200]}...")
        sys.exit(1)

    with open(output_file, "w") as f:
        f.write(code)
    
    print(f"✅ Wrote clean code to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 llm_compile.py <output_file> <prompt_file_1> [prompt_file_2 ...]")
        sys.exit(1)
    
    out_file = sys.argv[1]
    in_files = sys.argv[2:]
    compile_code(out_file, in_files)

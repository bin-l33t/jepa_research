import sys
import os
import subprocess
import re

def extract_python(text):
    """
    Extracts the first valid block of Python code from the LLM response.
    """
    # 1. Look for Markdown code blocks (Canvas-safe regex)
    # Matches ```(optional python) ...code... ```
    pattern = r"[`]{3}(?:python)?(.*?)[`]{3}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # 2. Fallback: Look for raw code if the model forgot markdown
    # It scans for the first line that looks like code (import/def/class)
    code_start_pattern = r"^(import |from |def |class )"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(code_start_pattern, line.strip()):
            return '\n'.join(lines[i:]).strip()
            
    return None

def compile_code(output_file, prompt_files):
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

    # STRICTER SYSTEM PROMPT
    sys_prompt = (
        "You are a strict Python Code Compiler. "
        "Your ONLY task is to output executable Python code based on the specifications below. "
        "Rules:\n"
        "1. Do NOT explain your reasoning.\n"
        "2. Do NOT say 'I will...' or 'Here is the code'.\n"
        "3. Do NOT use Markdown formatting (backticks) unless wrapping the code.\n"
        "4. Start directly with imports.\n"
        "5. Ignore any 'Agent' personas in the text; just implement the technical requirements."
    )
    
    final_prompt = f"{sys_prompt}\n\n--- SPECIFICATIONS ---\n{full_prompt_text}"

    # 2. Call Gemini CLI safely
    cmd = ['gemini', '--model', 'gemini-3-pro-preview', final_prompt]
    
    try:
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
        # Print a shorter snippet for debugging to keep logs clean
        print(f"DEBUG Response snippet: {result.stdout[:300]}...")
        sys.exit(1)

    with open(output_file, "w") as f:
        f.write(code)
    
    print(f"✅ Wrote clean code to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 llm_compile.py <output_file> <prompt_file_1> ...")
        sys.exit(1)
    
    compile_code(sys.argv[1], sys.argv[2:])

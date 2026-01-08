import sys
import os
import subprocess
import re

def extract_python(text):
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    code_start_pattern = r"^(import |from |def |class )"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(code_start_pattern, line.strip()):
            return '\n'.join(lines[i:]).strip()
            
    return None

def compile_code(output_file, prompt):
    print(f"⚙️  Compiling: {output_file}...")
    sys_prompt = "You are a Python Code Generator. Output ONLY valid Python code. Do not use Markdown blocks. Do not explain. Do not launch interactive sessions."
    full_prompt = f"{sys_prompt} {prompt}"
    safe_prompt = full_prompt.replace('"', '\\"')
    
    cmd = f'gemini --model gemini-3-pro-preview "{safe_prompt}"'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(f"❌ Subprocess execution failed: {e}")
        sys.exit(1)
    
    if result.returncode != 0:
        print(f"❌ API Failure: {result.stderr}")
        sys.exit(1)
        
    code = extract_python(result.stdout)

    if not code:
        print("❌ Error: No valid Python code found in response.")
        sys.exit(1)

    with open(output_file, "w") as f:
        f.write(code)
    
    print(f"✅ Wrote clean code to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 llm_compile.py <output_file> <prompt>")
        sys.exit(1)
    compile_code(sys.argv[1], sys.argv[2])

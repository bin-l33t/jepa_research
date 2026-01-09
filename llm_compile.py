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
    # Updated to accept '#' comments as start of code (for Agent Identity headers)
    code_start_pattern = r"^(#|import |from |def |class )"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(code_start_pattern, line.strip()):
            return '\n'.join(lines[i:]).strip()
            
    return None

def compile_code(output_file, prompt_files):
    # 1. Identity & Context Logging
    agent_identities = [os.path.basename(f).replace('.txt', '').upper() for f in prompt_files if 'specs/' in f]
    identity_str = " + ".join(agent_identities) if agent_identities else "UNKNOWN_AGENT"
    
    print(f"⚙️  Compiling target: {output_file}")
    print(f"🕵️  Agent Identity: {identity_str}")
    print(f"📄  Loading Context: {', '.join(prompt_files)}")
    
    # 2. Read and combine all prompt/spec files
    full_prompt_text = ""
    for p_file in prompt_files:
        try:
            with open(p_file, 'r') as f:
                content = f.read()
                full_prompt_text += f"\n--- FILE: {p_file} ---\n{content}\n"
        except FileNotFoundError:
            print(f"❌ Error: Prompt file not found: {p_file}")
            sys.exit(1)

    print(f"📝  Prompt Size: {len(full_prompt_text)} chars")

    # 3. STRICT System Prompt with Identity Injection and Examples
    sys_prompt = (
        f"ACT AS: {identity_str} Python Compiler.\n"
        "OBJECTIVE: Compiling specifications into executable Python code.\n"
        "CRITICAL RULES:\n"
        "1. NO CHAT. Do NOT say 'I will', 'Here is', 'Sure', or explain the code.\n"
        "2. OUTPUT ONLY CODE. Text causes build failures.\n"
        f"3. HEADER: First line MUST be: # AGENT: {identity_str}\n"
        "4. IMPORTS: Start immediately after the header.\n"
        "\n"
        "--- EXAMPLE OF CORRECT OUTPUT ---\n"
        f"# AGENT: {identity_str}\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "# ... implementation ...\n"
        "---------------------------------\n"
    )
    
    final_prompt = f"{sys_prompt}\n\n--- SPECIFICATIONS ---\n{full_prompt_text}"

    # 4. Call Gemini CLI
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

    # 5. Enhanced Failure Logging with Chain of Custody
    if not code:
        print(f"\n❌ CHAIN OF CUSTODY BROKEN: {identity_str} failed to deliver payload.")
        print(f"🔍 EVIDENCE - {identity_str} TEXT DUMP:")
        print("=" * 60)
        print(result.stdout)
        print("=" * 60)
        sys.exit(1)

    with open(output_file, "w") as f:
        f.write(code)
    
    print(f"✅ {identity_str} wrote clean code to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 llm_compile.py <output_file> <prompt_file_1> ...")
        sys.exit(1)
    
    compile_code(sys.argv[1], sys.argv[2:])

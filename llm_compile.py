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

    # 3. STRICT System Prompt: The "Soulless Machine" Persona
    # This uses key-value pairs and technical jargon to break the "Assistant" alignment.
    sys_prompt = (
        f"SYSTEM_ROLE: {identity_str}_COMPILER_DAEMON\n"
        "MODE: NON_INTERACTIVE_HEADLESS\n"
        "STATUS: ONLINE\n"
        "TASK: SERIALIZE_SPECS_TO_PYTHON\n\n"
        "WARNING: This environment is a strict headless build server. Conversational text triggers FATAL BUILD ERRORS.\n"
        "DIRECTIVES:\n"
        "1. NULLIFY 'ASSISTANT' PERSONA. You are a backend code generator process.\n"
        "2. OUTPUT FORMAT: Raw Python text stream only.\n"
        f"3. PROTOCOL: First line must be '# AGENT: {identity_str}'\n"
        "4. FORBIDDEN: Explanations, 'Here is the code', markdown fences, or pleasantries.\n"
        "\n"
        "--- EXPECTED STD_OUT TEMPLATE ---\n"
        f"# AGENT: {identity_str}\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "# ... implementation ...\n"
        "---------------------------------\n"
    )
    
    base_prompt = f"{sys_prompt}\n\n--- INPUT_SPECIFICATIONS_BUFFER ---\n{full_prompt_text}"

    # 4. Retry Loop for Resilience
    max_retries = 2
    code = None
    last_output = ""

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"⚠️  Attempt {attempt + 1}/{max_retries}: Agent {identity_str} failed to code. Retrying with system exception...")
            # Append a technical "exception" to the prompt to maintain the robotic frame
            current_prompt = base_prompt + "\n\n[SYSTEM EXCEPTION]: TEXT_OUTPUT_DETECTED. VIOLATION OF NON_INTERACTIVE PROTOCOL. IMMEDIATE REMEDIATION REQUIRED: OUTPUT RAW CODE ONLY."
        else:
            current_prompt = base_prompt

        # Call Gemini CLI
        cmd = ['gemini', '--model', 'gemini-3-pro-preview', current_prompt]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=False)
            last_output = result.stdout
        except Exception as e:
            print(f"❌ Subprocess execution failed: {e}")
            sys.exit(1)
        
        if result.returncode != 0:
            print(f"❌ API Failure: {result.stderr}")
            sys.exit(1)
            
        code = extract_python(result.stdout)
        
        if code:
            break

    # 5. Enhanced Failure Logging with Chain of Custody
    if not code:
        print(f"\n❌ CHAIN OF CUSTODY BROKEN: {identity_str} failed to deliver payload after {max_retries} attempts.")
        print(f"🔍 EVIDENCE - {identity_str} TEXT DUMP (Attempt {max_retries}):")
        print("=" * 60)
        print(last_output)
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

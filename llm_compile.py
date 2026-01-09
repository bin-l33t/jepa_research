import sys
import os
import subprocess
import re
import datetime

LOG_FILE = "build.log"
MODEL_NAME = "gemini-3-pro-preview"

def log(message):
    """Prints to console and appends to a log file."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    # Print to console
    print(message)
    # Append to log file with flush to ensure persistence
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()

def extract_python(text):
    """
    Extracts valid Python code.
    Prioritizes Markdown blocks, falls back to raw code detection.
    """
    # 1. Look for Markdown code blocks
    pattern = r"[`]{3}(?:python)?(.*?)[`]{3}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # 2. Fallback: Look for raw code
    # We look for the Agent Identity Header or standard imports
    lines = text.split('\n')
    start_index = -1
    
    # Scan for the start of code
    code_indicators = [r"^# AGENT:", r"^import ", r"^from ", r"^class ", r"^def "]
    
    for i, line in enumerate(lines):
        for indicator in code_indicators:
            if re.match(indicator, line.strip()):
                start_index = i
                break
        if start_index != -1:
            break
            
    if start_index != -1:
        return '\n'.join(lines[start_index:]).strip()
            
    return None

def clean_code_artifact(code):
    """
    Cleans up the generated code artifact:
    1. Removes the PROMPT/CONTEXT docstring block to keep the file clean.
    2. Removes any trailing prompt separators (e.g. ----------------).
    """
    # Remove the massive docstring block we injected
    # Matches """\nSPECIFICATIONS AND CONTEXT: ... """
    code = re.sub(r'"""\nSPECIFICATIONS AND CONTEXT:.*?"""', '', code, flags=re.DOTALL)
    
    # Remove the trailing separator line if it exists
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        if "---------------------------------" in line:
            continue
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines).strip()

def compile_code(output_file, prompt_files):
    os.makedirs("logs", exist_ok=True)
    
    # 1. Identity & Context
    agent_identities = [os.path.basename(f).replace('.txt', '').upper() for f in prompt_files if 'specs/' in f]
    identity_str = " + ".join(agent_identities) if agent_identities else "UNKNOWN_AGENT"
    
    log(f"----------------------------------------------------------------")
    log(f"⚙️  Compiling target: {output_file}")
    log(f"🤖  Model: {MODEL_NAME}")
    log(f"🕵️  Agent: {identity_str}")
    
    # 2. Read Context
    full_prompt_text = ""
    for p_file in prompt_files:
        try:
            with open(p_file, 'r') as f:
                content = f.read()
                full_prompt_text += f"\n--- FILE: {p_file} ---\n{content}\n"
        except FileNotFoundError:
            log(f"❌ Error: Prompt file not found: {p_file}")
            sys.exit(1)

    # 3. CONSTRUCT THE "PARTIAL FILE" TEMPLATE
    # This is the "Autocomplete Hack". We embed specs in a docstring.
    code_template = f'''# AGENT: {identity_str}
"""
SPECIFICATIONS AND CONTEXT:
{full_prompt_text}
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
# (Add other likely imports based on context)
import math

# IMPLEMENTATION STARTS HERE
'''

    final_prompt = (
        "TASK: COMPLETE THE PYTHON FILE BELOW.\n"
        "INSTRUCTIONS:\n"
        "1. The file has already started. Output the REST of the code.\n"
        "2. You MAY repeat the imports if you wish, or start defining classes/functions immediately.\n"
        "3. DO NOT CONVERSE. OUTPUT CODE ONLY.\n"
        "\n"
        "--- START OF PYTHON FILE ---\n"
        f"{code_template}"
    )

    # 4. Retry Loop
    max_retries = 2
    code = None
    last_output = ""
    last_stderr = ""

    for attempt in range(max_retries):
        current_prompt = final_prompt
        
        if attempt > 0:
            log(f"⚠️  Attempt {attempt + 1}/{max_retries}: Agent {identity_str} failed. escalating...")
            # If it failed, we prepend a directive to the top
            current_prompt = "!!! CRITICAL: OUTPUT ONLY VALID PYTHON CODE. NO TEXT. !!!\n\n" + final_prompt

        # Save prompt
        log_base_name = os.path.join("logs", f"{output_file}.attempt_{attempt+1}")
        with open(f"{log_base_name}.prompt", "w") as f:
            f.write(current_prompt)

        cmd = ['gemini', '--model', MODEL_NAME, current_prompt]
        log(f"🔌  Invoking: gemini [PROMPT_SIZE_{len(current_prompt)}B]")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=False)
            last_output = result.stdout
            last_stderr = result.stderr
            
            # Save response
            with open(f"{log_base_name}.response", "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n--- STDERR ---\n")
                    f.write(result.stderr)
                    
        except Exception as e:
            log(f"❌ Subprocess failed: {e}")
            sys.exit(1)
        
        code = extract_python(result.stdout)
        if code:
            break

    # 5. Failure Handling
    if not code:
        log(f"\n❌ CHAIN OF CUSTODY BROKEN: {identity_str} failed to code.")
        log(f"🔍 DUMPING LAST RESPONSE:")
        log("=" * 60)
        log(last_output if last_output.strip() else last_stderr)
        log("=" * 60)
        sys.exit(1)

    # 6. Success - Clean and Save
    # We clean the artifact to remove the massive prompt injection and invalid separators
    final_code = clean_code_artifact(code)
    
    # Ensure the Header exists if it was stripped
    if "# AGENT:" not in final_code:
        final_code = f"# AGENT: {identity_str}\n{final_code}"

    with open(output_file, "w") as f:
        f.write(final_code)
    
    log(f"✅ {identity_str} generated: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 llm_compile.py <output_file> <prompt_file_1> ...")
        sys.exit(1)
    
    compile_code(sys.argv[1], sys.argv[2:])

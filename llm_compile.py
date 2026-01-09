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
    # Print with emojis to console
    print(message)
    # Strip emojis for cleaner log file
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

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
    # Ensure logs directory exists for individual agent artifacts
    os.makedirs("logs", exist_ok=True)
    
    # 1. Identity & Context Logging
    agent_identities = [os.path.basename(f).replace('.txt', '').upper() for f in prompt_files if 'specs/' in f]
    identity_str = " + ".join(agent_identities) if agent_identities else "UNKNOWN_AGENT"
    
    log(f"----------------------------------------------------------------")
    log(f"⚙️  Compiling target: {output_file}")
    log(f"🤖  Model: {MODEL_NAME}")
    log(f"🕵️  Agent: {identity_str}")
    log(f"📄  Context: {', '.join(prompt_files)}")
    
    # 2. Read and combine all prompt/spec files
    full_prompt_text = ""
    for p_file in prompt_files:
        try:
            with open(p_file, 'r') as f:
                content = f.read()
                full_prompt_text += f"\n--- FILE: {p_file} ---\n{content}\n"
        except FileNotFoundError:
            log(f"❌ Error: Prompt file not found: {p_file}")
            sys.exit(1)

    log(f"📝  Prompt Size: {len(full_prompt_text)} chars")

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
            log(f"⚠️  Attempt {attempt + 1}/{max_retries}: Agent {identity_str} failed to code. Retrying with system exception...")
            # Append a technical "exception" to the prompt to maintain the robotic frame
            current_prompt = base_prompt + "\n\n[SYSTEM EXCEPTION]: TEXT_OUTPUT_DETECTED. VIOLATION OF NON_INTERACTIVE PROTOCOL. IMMEDIATE REMEDIATION REQUIRED: OUTPUT RAW CODE ONLY."
        else:
            current_prompt = base_prompt

        # Save the exact prompt to a log file for user inspection
        log_base_name = os.path.join("logs", f"{output_file}.attempt_{attempt+1}")
        with open(f"{log_base_name}.prompt", "w") as f:
            f.write(current_prompt)

        # Call Gemini CLI
        cmd = ['gemini', '--model', MODEL_NAME, current_prompt]
        
        # Log the CLI invocation
        log(f"🔌  Invoking: gemini --model {MODEL_NAME} [PROMPT_PAYLOAD_{len(current_prompt)}B]")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=False)
            last_output = result.stdout
            
            # Save the raw response to a log file for user inspection
            with open(f"{log_base_name}.response", "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n--- STDERR ---\n")
                    f.write(result.stderr)
                    
        except Exception as e:
            log(f"❌ Subprocess execution failed: {e}")
            sys.exit(1)
        
        if result.returncode != 0:
            log(f"❌ API Failure (Exit Code {result.returncode}): {result.stderr}")
            sys.exit(1)
            
        code = extract_python(result.stdout)
        
        if code:
            break

    # 5. Enhanced Failure Logging with Chain of Custody
    if not code:
        log(f"\n❌ CHAIN OF CUSTODY BROKEN: {identity_str} failed to deliver payload after {max_retries} attempts.")
        log(f"🔍 EVIDENCE - {identity_str} TEXT DUMP (Attempt {max_retries}):")
        log("=" * 60)
        log(last_output)
        log("=" * 60)
        sys.exit(1)

    with open(output_file, "w") as f:
        f.write(code)
    
    log(f"✅ {identity_str} wrote clean code to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 llm_compile.py <output_file> <prompt_file_1> ...")
        sys.exit(1)
    
    compile_code(sys.argv[1], sys.argv[2:])

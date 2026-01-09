SHELL := /bin/bash

# --- THE ENGINE ---
COMPILER = python3 llm_compile.py

# --- THE PIPELINE ---
all: report

# STEP 1: ARCHITECT (Core Library)
# Notice: We pass the FILE PATHS now, not string prompts
jepa_core.py: specs/architect.txt specs/math_context.txt
	@echo "🏗️  Agent 1: Architecting Core Library..."
	@$(COMPILER) jepa_core.py specs/architect.txt specs/math_context.txt

# STEP 2: VALIDATOR (Unit Tests)
test_jepa.py: jepa_core.py specs/validator.txt
	@echo "🧪 Agent 1b: Writing Unit Tests..."
	@$(COMPILER) test_jepa.py specs/validator.txt

# STEP 3: THE BARRIER (Execution)
test_passed: test_jepa.py
	@echo "🛡️  Running Unit Tests..."
	@python3 test_jepa.py && touch test_passed

# STEP 4: SCIENTIST (Training Script)
train_vector.py: test_passed specs/scientist.txt
	@echo "🔬 Agent 2: Writing Training Script..."
	@$(COMPILER) train_vector.py specs/scientist.txt

# STEP 5: EXECUTION
run_experiment: train_vector.py
	@echo "🚀 Launching Verified Experiment..."
	@python3 train_vector.py

# STEP 6: OVERSEER (Analysis)
report: run_experiment
	@echo "🤖 Overseer: Analyzing Results..."
	@python3 analyze_results.py

# --- UTILITIES ---
clean:
	rm -f jepa_core.py test_jepa.py train_vector.py test_passed metrics.csv covariance.png REPORT.txt
	rm -rf __pycache__

graph:
	@echo "Core -> Tests -> (Barrier) -> Training -> Analysis"

SHELL := /bin/bash

# CONFIGURATION
COMPILER = python3 llm_compile.py

# THEORETICAL CONTEXT (Injected into Agents)
# Using single quotes for the variable definition to treat it as a literal string
MATH_SPECS = 'Implement SIGRegLoss and VectorFlow. CRITICAL: In SIGRegLoss, you MUST normalize the random projection matrix A so columns have norm=1 (A = A / A.norm(dim=0)).'

# PIPELINE TARGETS
all: report

# --- STEP 1: ARCHITECT ---
jepa_core.py:
	@echo "🏗️  Agent 1: Architecting Core Library..."
	@$(COMPILER) jepa_core.py "Write a PyTorch library with SIGRegLoss and VectorFlow. $(MATH_SPECS). Use standard torch modules."

# --- STEP 2: VALIDATOR ---
test_jepa.py: jepa_core.py
	@echo "🧪 Agent 1b: Writing Unit Tests..."
	@$(COMPILER) test_jepa.py "Read jepa_core.py. Write a unit test script. 1. Test SIGRegLoss on Gaussian input. 2. Test SIGRegLoss on Uniform input. 3. Check VectorFlow shapes. Import from jepa_core."

# --- STEP 3: THE BARRIER ---
test_passed: test_jepa.py
	@echo "🛡️  Running Unit Tests..."
	@python3 test_jepa.py && touch test_passed

# --- STEP 4: SCIENTIST ---
train_vector.py: test_passed
	@echo "🔬 Agent 2: Writing Training Script..."
	@$(COMPILER) train_vector.py "Context: jepa_core.py is valid. Task: Write a training script for CIFAR-10. Batch=128. Loop: 5 Epochs. Steps per batch: 10. Loss: Agreement + 0.1*SIGReg. Optimizer: Adam(1e-3). LOGGING: Print Epoch, Step, Loss, Agreement, SIGReg, Acc every 50 steps. Save metrics to metrics.csv. Save covariance.png. Wrap main logic in if __name__ == '__main__': train(). CRITICAL: OUTPUT PURE CODE wrapped in markdown python blocks."

# --- STEP 5: EXECUTION ---
run_experiment: train_vector.py
	@echo "🚀 Launching Verified Experiment..."
	@python3 train_vector.py

# --- STEP 6: OVERSEER ---
report: run_experiment
	@echo "🤖 Overseer: Analyzing Results..."
	@python3 analyze_results.py

clean:
	rm -f jepa_core.py test_jepa.py train_vector.py test_passed metrics.csv covariance.png REPORT.txt

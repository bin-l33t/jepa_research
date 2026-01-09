import pandas as pd
import sys
import os

def analyze():
    if not os.path.exists('metrics.csv'):
        print("❌ CRITICAL: No metrics found.")
        sys.exit(1)

    try:
        df = pd.read_csv('metrics.csv')
        # Normalize columns if needed
        if 'loss' not in df.columns:
             # Fallback attempt to name columns if header is missing
            if len(df.columns) == 6:
                df.columns = ['epoch', 'step', 'loss', 'agreement', 'sigreg', 'acc']
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)

    if len(df) < 5:
        print("❌ Insufficient data points.")
        sys.exit(1)

    final_agreement = df['Agreement'].iloc[-1] if 'Agreement' in df.columns else df.iloc[-1, 3]
    final_sigreg = df['SIGReg'].iloc[-1] if 'SIGReg' in df.columns else df.iloc[-1, 4]
    
    # Logic for Success/Failure
    status = "SUCCESS"
    if final_agreement > 1.0:
        status = "FAILURE (DRIFT - Agreement too high)"
    elif final_sigreg > 5.0: # Relaxed threshold
        status = "FAILURE (NON-ISOTROPIC - Reg too high)"

    with open("REPORT.txt", "w") as f:
        f.write("--- AUTOMATED EXPERIMENT REPORT ---\n")
        f.write(f"STATUS: {status}\n")
        f.write(f"FINAL AGREEMENT: {final_agreement:.4f}\n")
        f.write(f"FINAL SIGREG: {final_sigreg:.4f}\n")

    print(open("REPORT.txt").read())
    
    if "FAILURE" in status:
        sys.exit(1)

if __name__ == "__main__":
    analyze()

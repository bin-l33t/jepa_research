import pandas as pd
import sys
import os

def analyze():
    if not os.path.exists('metrics.csv'):
        print("❌ CRITICAL: No metrics found.")
        sys.exit(1)

    try:
        df = pd.read_csv('metrics.csv')
        if 'loss' not in df.columns:
            df.columns = ['epoch', 'step', 'loss', 'agreement', 'sigreg', 'acc']
    except:
        sys.exit(1)

    if len(df) < 5:
        sys.exit(1)

    start_loss = df['loss'].iloc[0]
    end_loss = df['loss'].iloc[-1]
    final_sigreg = df['sigreg'].iloc[-1]
    final_agreement = df['agreement'].iloc[-1]
    final_acc = df['acc'].iloc[-1]

    with open("REPORT.txt", "w") as f:
        f.write("--- AUTOMATED EXPERIMENT REPORT ---\n")
        status = "UNKNOWN"
        
        if final_agreement > 1.0:
            status = "FAILURE (DRIFT)"
        elif final_sigreg > 1.5:
            status = "FAILURE (NON-ISOTROPIC)"
        elif final_acc < 0.20:
            status = "FAILURE (BLIND)"
        else:
            status = "SUCCESS"

        f.write(f"STATUS: {status}\n")
        f.write(f"AGREEMENT: {final_agreement:.4f}\n")
        f.write(f"ACCURACY: {final_acc:.1%}\n")

    print(open("REPORT.txt").read())
    
    if "FAILURE" in status:
        sys.exit(1)

if __name__ == "__main__":
    analyze()

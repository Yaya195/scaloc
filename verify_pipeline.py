"""
Quick pipeline verification test
"""
import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Show output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ FAILED with exit code {result.returncode}")
        return False
    else:
        print(f"✓ SUCCESS")
        return True

def main():
    print("="*70)
    print("DATA PIPELINE VERIFICATION TEST")
    print("="*70)
    
    # Test each step
    steps = [
        ("python -m src.data.clean_preprocess", "Step 1: Clean & preprocess"),
        ("python -m src.data.build_domains", "Step 2: Build domains"),
        ("python -m src.data.build_rps", "Step 3: Build RPs"),
        ("python -m src.data.build_graphs", "Step 4: Build graphs"),
        ("python -m src.data.build_samples", "Step 5: Build samples"),
    ]
    
    results = []
    for cmd, desc in steps:
        success = run_command(cmd, desc)
        results.append((desc, success))
        if not success:
            print(f"\n❌ Pipeline failed at: {desc}")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("PIPELINE VERIFICATION SUMMARY")
    print("="*70)
    for desc, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {desc}")
    
    print("\n" + "="*70)
    print("✓ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Quick data check
    print("\nVerifying output files exist...")
    from pathlib import Path
    
    checks = [
        ("data/interim/clean/train_clean.parquet", "Training data cleaned"),
        ("data/interim/clean/val_clean.parquet", "Validation data cleaned"),
        ("data/processed/samples/train_samples.parquet", "Training samples"),
        ("data/processed/samples/val_samples.parquet", "Validation samples"),
    ]
    
    for path, desc in checks:
        exists = Path(path).exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {desc}: {path}")
    
    print("\n✓ Pipeline verification complete!")

if __name__ == "__main__":
    main()

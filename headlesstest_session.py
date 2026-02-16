from zfisher.core import session
from pathlib import Path

# 1. Setup paths
# Use the same test paths you used in the widget
out_dir = Path.home() / "zFISHer_Headless_Test"
r1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2")
r2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2")

# 2. Define a simple terminal callback for progress
def term_progress(p, text):
    print(f"[{p}%] {text}")

# 3. Execute the headless initialization
success = session.initialize_new_session(
    output_dir=out_dir, 
    r1_path=r1, 
    r2_path=r2,
    progress_callback=term_progress
)

if success:
    print(f"\n✅ Headless Session Initialized at: {out_dir}")
else:
    print("\n❌ Failed: Session already exists or path error.")
import pexpect
import sys
import time

def main():
    print("Starting Wisdom Engine...")
    engine = pexpect.spawn("./target/debug/wisdom", encoding='utf-8')
    
    # 1. Initialize UCCI
    print("-> ucci")
    engine.sendline("ucci")
    engine.expect("ucciok", timeout=5)
    print("<- " + engine.before + "ucciok")

    # 2. Check readiness
    print("-> isready")
    engine.sendline("isready")
    engine.expect("readyok", timeout=5)
    print("<- readyok")

    # 3. Load fen
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    print(f"-> position fen {fen}")
    engine.sendline(f"position fen {fen}")
    
    # Check evaluation
    print("-> eval")
    engine.sendline("eval")
    engine.expect("info eval", timeout=2)
    print("<- " + engine.before + "info eval")

    # 4. Search for depth 4
    print("-> go depth 4")
    engine.sendline("go depth 4")
    
    engine.expect("bestmove", timeout=10)
    best_move = engine.readline().strip()
    print(f"<- bestmove {best_move}")

    print("-> quit")
    engine.sendline("quit")

if __name__ == "__main__":
    main()

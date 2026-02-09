import argparse
from data.latte_recon_reader import LatteReconReader, LatteReconLayout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=str, required=True)
    ap.add_argument("--ball_row", type=int, default=-999)
    ap.add_argument("--hit_row", type=int, default=-999)
    args = ap.parse_args()

    layout = LatteReconLayout()
    if args.ball_row != -999:
        layout.ball_row = args.ball_row
    if args.hit_row != -999:
        layout.hit_row = args.hit_row

    reader = LatteReconReader(layout=layout, verbose=True)
    out = reader.load(args.recon)

    ball = out["ball"]
    print("\n[Ball stats]")
    print("  first:", ball[0])
    print("  last :", ball[-1])
    print("  min  :", ball.min(axis=0))
    print("  max  :", ball.max(axis=0))

    if out["hit"] is not None:
        hit = out["hit"]
        p1 = (hit[:, 0] > 0.5).sum()
        p2 = (hit[:, 1] > 0.5).sum()
        tb = (hit[:, 2] > 0.5).sum()
        print("\n[Hit stats]")
        print(f"  p1_hit_frames={p1}, p2_hit_frames={p2}, table_hit_frames={tb}")
    else:
        print("\n[Hit stats] hit_row not found / not used")

if __name__ == "__main__":
    main()

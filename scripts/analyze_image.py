import argparse
from ringfit.fitting import analyze_image
from ringfit import extraction as ex            # NEW
import matplotlib.pyplot as plt                 # NEW

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a photon-ring image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="RBP threshold")
    parser.add_argument("--radius", type=float, default=30,
                        help="RBP masking radius")
    parser.add_argument("--margin", type=int,   default=15,
                        help="Ignore edge margin (px)")
    parser.add_argument("--show", action="store_true",            # NEW
                        help="Display the image with centres/points")
    args = parser.parse_args()

    analyze_image(im=args.image,
                  threshold=args.threshold,
                  radius=args.radius,
                  margin=args.margin)

    if args.show:                                                 # NEW
        ex.show_centers(args.image)                               # NEW
        plt.show()                                                # NEW

if __name__ == "__main__":
    main()

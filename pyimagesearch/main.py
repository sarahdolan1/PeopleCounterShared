import argparse
import readImage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="path to file")
    args = parser.parse_args()
    video = readImage.readVideo(args.filepath)
    while True:
        frame = readImage.readFrame(video)
        readImage.displayFrame(frame)


if __name__ == "__main__":
    main()

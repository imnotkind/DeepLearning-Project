import environment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="An absolute URL for rendering")
    args = parser.parse_args()

    url = args.url

    cenv = environment.ChessEnvironment()
    print(cenv.pc) #player color

    str = input()
    assert(str == "ok")
    cenv.process(url=url)

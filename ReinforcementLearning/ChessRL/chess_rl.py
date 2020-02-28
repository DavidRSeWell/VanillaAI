import chess
import chess.svg
import matplotlib.pyplot as plt


board = chess.Board()

if __name__ == '__main__':

    board = chess.Board()

    board_svg = chess.svg.board()

    plt.savefig(board_svg,)
    print('done running chess')

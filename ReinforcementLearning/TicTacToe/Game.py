import numpy as np
import itertools

class RandomPlayer:

    def __init__(self,side='X'):
        self.current_state = None # the game expects a player to have this attribute
        self.side = side

    def __repr__(self):
        return self.side

    def __str__(self):
        return self.side

    def get_legal_moves(self,board):
        '''
        Get all the possible moves that
        could be played from the current board
        :return:
        '''
        legal_moves = []
        for s in board.keys():
            if board[s] == ' ':
                legal_moves.append(s)

        return legal_moves

    def act(self,board):
        '''
        Takes in the boards state
        and just returns a random action
        :param board:
        :return:
        '''

        legal_moves = self.get_legal_moves(board)

        return (self.side,np.random.choice(legal_moves))

class TabularQPlayer:
    '''
    Im not sophistaced but I work
    '''

    def __init__(self,side='X',epsilon=0.001):

        self.current_state = None # varaible used for tracking state
        self.epsilon = epsilon
        self.side = side

        self.Q = None # Q matrix with rows = states and columns = actions
        self.S = None # matrix of all possible states

    def act(self, board):
        '''
        Takes in the boards state
        and just returns a random action
        :param board: dictionary
        :return:
        '''

        legal_moves = self.get_legal_moves(board)

        return self.e_greedy(legal_moves)

    def e_greedy(self,moves):
        '''
        Take e-greedy action
        :param moves: possible legal actions to take
        :return:
        '''

        if np.random.random() < self.epsilon:
            # take random action
            return np.random.choice(moves)

        else:
            return moves[np.argmax(self.Q[moves])] # return the greedy action

    def get_greedy(self,s):
        '''
        Return the greedy action for a given state
        :param s:
        :return:
        '''

        s_index = self.get_state_index(s)

        legal_moves = np.where(s == 0)[0]

        return np.argmax(self.Q[s_index][legal_moves])

    def get_legal_moves(self,board):
        '''
        Get all the possible moves that
        could be played from the current board
        :return: list of integers
        '''

        legal_moves = []
        i = 0
        for s in board.keys():
            if board[s] == ' ':
                legal_moves.append(i)
            i += 1

        return legal_moves

    def get_state_index(self,s):
        '''
        Look up the index of the current state
        :param s:
        :return:
        '''
        return np.where((self.S == s).all(axis=1))[0][0]

    def init(self):

        self.S = np.array([np.array(i) for i in itertools.product([0,1,-1], repeat = 9)]) # create all possible combinations of states
        self.Q = np.zeros((len(self.S),9)) # Q matrix of values

    def update(self,a,r,s):
        '''

        :param reward:
        :param s1: current state
        :param s2: next state
        :return:
        '''

        s1_index = self.get_state_index(self.current_state)

        s2_index = self.get_state_index(self.current_state)
        s2 = self.S[s2_index]

        max_q_next = self.Q[s2_index][self.get_greedy(s2)]

        self.Q[s1_index,a] = self.Q[s1_index,a] + self.alpha(r + self.gamma*max_q_next - self.Q[s1_index,a])

class TicTacToe:
    '''
    This class handles all the logic of running a tic tac
    toe game. It expects as input two players which themselves
    are expected to have an 'act' method that will return the
    action the player wants to take. The players can either
    be bots or humans
    '''

    def __init__(self,player1=None,player2=None):

        self.board = '''
                | {s1} | {s2} | {s3} |    
                 ------------
                | {s4} | {s5} | {s6} |    
                 ------------
                | {s7} | {s8} | {s9} |  
            '''

        self.pieces = {' ': 0, 'X': 1, 'O': -1} # used in converting the board dictionary to an array
        self.player1 = player1
        self.player2 = player2
        self.board_state = {'s1':' ','s2':' ','s3':' '
                        ,'s4':' ','s5':' ','s6':' ',
                           's7':' ','s8':' ','s9':' '}

        self.game_state = {
            'player':self.player1, # whos turn is it anyways?
            'win_state': 0 # 0 if game is still going 1 if player 'X' won and -1 if player 'O' won 2 for a draw
        }

    def __str__(self):

        return self.board.format(** self.board_state)

    def convert_board_to_array(self):
        '''
        Converts the board dictionary to
        a numpy array
        :param board_state: dictionary
        :return:
        '''

        board = np.zeros((9,))
        i = 0
        for k, v in self.board_state.items():
            board[i] = self.pieces[v]
            i += 1

        return np.reshape(board, (3, 3))

    def is_legal(self,action):
        '''
        Checks to make sure an action is legally possible
        :return:
        '''
        player , coord = action

        if self.game_state['player'].side != player:
            print("Its not player {}'s turn".format(player))
            return False
        elif self.board_state[coord] != ' ': # if someone has already played there
            print("Someone has already played {}".format(coord))
            return False

        elif coord not in self.board_state:
            print("Coordinate {} is not on the board".format(coord))
            return False

        else:
            return True

    def is_draw(self):

        board_mat = self.convert_board_to_array() # look at board as an array

        zero_count = len(np.where(board_mat == 0))

        if zero_count == 0:
            return True
        else:
            return False

    def is_win(self):
        '''
        Check if the current board is a
        win for a given player or if the
        game is a draw.

        :return:
        '''

        board_mat = self.convert_board_to_array() # look at board as an array

        diag_1 = np.abs(sum([board_mat[i][i] for i in range(3)]))  # check diagnoal
        diag_2 = np.abs(sum([board_mat[i][3 - i - 1] for i in range(3)]))  # check other diagnol

        if 3 in np.abs(board_mat.sum(axis=1)): # row sum
            return True

        elif 3 in np.abs(board_mat.sum(axis=0)): # col sum
            return True

        elif (3 in (diag_1,diag_2)):
            return True

        else:
            return False


    def run(self,show_board=True):
        '''
        The main method used to simulate a single game
        :return:
        '''

        self.player1.current_state = self.board_state

        self.player2.current_state = self.board_state

        while self.game_state['win_state'] == 0:

            action = self.game_state['player'].act(self.board_state)

            r = self.update(action)

            self.game_state['player'].update(action[1],r,self.board_state)

            self.player1.current_state = self.board_state

            self.player2.current_state = self.board_state

            if show_board:
                print(self)

    def update(self,action):
        '''
        Take in an action from a player
        and update the game state
        :param action: tuple (player, board coordinate) i.e. ('X', s1)
        :return:
        '''

        if self.is_legal(action):
            # update the board
            self.board_state[action[1]] = self.game_state['player'].side

            # check if this is a winning move
            if self.is_win():
                self.game_state['win_state'] = self.game_state['player'].side
                return 1

            elif self.is_draw():
                self.game_state['win_state'] = 2
                return -1

            # now update the current player
            if self.game_state['player'] == self.player1:

                self.game_state['player'] = self.player2
            else:

                self.game_state['player'] = self.player1

            return 0

        else:
            print('The action {} by player {} not legal!!!'.format(action[0],action[1]))
            return -1









if __name__ == '__main__':

    player1 = TabularQPlayer(side='X')
    player1.init()
    player2 = RandomPlayer(side='O')

    tic_tac_toe = TicTacToe(player1,player2)

    tic_tac_toe.run()



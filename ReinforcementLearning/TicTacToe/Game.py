import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns; sns.set()

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
        :return: list of integers
        '''

        legal_moves = []
        i = 0
        for s in board.keys():
            if board[s] == ' ':
                legal_moves.append(i)
            i += 1

        return legal_moves

    def update(self,r,s):
        '''

        :param reward:
        :param s1: current state
        :param s2: next state
        :return:
        '''
        pass

    def act(self,board):
        '''
        Takes in the boards state
        and just returns a random action
        :param board:
        :return:
        '''

        legal_moves = self.get_legal_moves(board)

        return np.random.choice(legal_moves)

class TabularQPlayer:
    '''
    Im not sophistaced but I work
    '''

    def __init__(self,side='X',epsilon=0.5):

        self.alpha = 0.1
        self.current_action = None
        self.current_state = None # varaible used for tracking state
        self.delta = 0.0001
        self.epsilon = epsilon
        self.gamma = 0.9
        self.pieces = {' ': 0, 'X': 1, 'O': -1} # used in converting the board dictionary to an array
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

    def convert_board_to_array(self,s):
        '''
        Converts the board dictionary to
        a numpy array
        :param board_state: dictionary
        :return:
        '''

        board = np.zeros((9,))
        i = 0
        for k, v in s.items():
            board[i] = self.pieces[v]
            i += 1

        return board

    def e_greedy(self,moves):
        '''
        Take e-greedy action
        :param moves: possible legal actions to take
        :return:
        '''

        s = self.convert_board_to_array(self.current_state)
        s_index = self.get_state_index(s)

        if np.random.random() < self.epsilon:
            # take random action
            return np.random.choice(moves)

        else:
            return moves[np.argmax(self.Q[s_index][moves])] # return the greedy action

    def get_greedy(self,s):
        '''
        Return the greedy action for a given state
        :param s:
        :return:
        '''

        s_index = self.get_state_index(s)

        legal_moves = np.where(s == 0)[0]

        if len(legal_moves) == 0:
            return False

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
        self.Q = np.random.random((len(self.S),9)) # Q matrix of values

    def update(self,r,s):
        '''

        :param reward:
        :param s1: current state
        :param s2: next state
        :return:
        '''

        s1 = self.convert_board_to_array(self.current_state)

        s1_index = self.get_state_index(s1)

        s2 = self.convert_board_to_array(s)

        s2_index = self.get_state_index(s2)

        max_action = self.get_greedy(s2)

        a = self.current_action

        if max_action:
            max_q_next = self.Q[s2_index][self.get_greedy(s2)]
            self.Q[s1_index, a] = self.Q[s1_index, a] + self.alpha * (r + self.gamma * max_q_next - self.Q[s1_index, a])


        else:
            # at a leaf node
            self.Q[s1_index,a] = self.Q[s1_index,a] + self.alpha*(r)

    def visualize_Q(self):
        '''
        Helper function to look at the average value for each square
        :return:
        '''
        v_mat = np.zeros((9,))

        for i in range(9):
            # first only look at times
            v_index = np.where(self.S[:,i] == 1)[0]
            v_mat[i] = self.Q[v_index].mean()

        v_mat = v_mat.reshape((3,3))

        sns.heatmap(v_mat)

        plt.show()

        for n in range(10):
            print()

class TicTacToe:
    '''
    This class handles all the logic of running a tic tac
    toe game. It expects as input two players which themselves
    are expected to have an 'act' method that will return the
    action the player wants to take. The players can either
    be bots or humans
    '''

    def __init__(self,player1=None,player2=None,base_player=None):

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

        return board

    def is_legal(self,action):
        '''
        Checks to make sure an action is legally possible
        :return:
        '''

        legal_moves = np.where(self.convert_board_to_array() == 0.0)[0]
        if action not in legal_moves:
            print('Action {} is not legal action'.format(action))
            print('CURRENT BOARD')
            print(self)
            return False

        else:
            return True

    def is_draw(self):

        board_mat = self.convert_board_to_array() # look at board as an array

        zero_count = len(np.where(board_mat == 0)[0])

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
        board_mat = np.reshape(board_mat,(3,3))

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

    def reset(self):
        '''
        Reset the board
        :return:
        '''
        self.board_state = {'s1': ' ', 's2': ' ', 's3': ' '
            , 's4': ' ', 's5': ' ', 's6': ' ',
                            's7': ' ', 's8': ' ', 's9': ' '}

        self.game_state = {
            'player': self.player1,  # whos turn is it anyways?
            'win_state': 0  # 0 if game is still going 1 if player 'X' won and -1 if player 'O' won 2 for a draw
        }

        self.player1.current_state = self.board_state.copy()
        self.player2.current_state = self.board_state.copy()

    def run(self,iters=100,simulate_iters=10,show_board=True):
        '''
        The main method used to simulate a single game

        :param iters:
        :param simulate_iters:
        :param show_board: True means print out the board display each update
        :param simulate: Set to True to not update the players
        :return:
        '''

        self.player1.current_state = self.board_state.copy()

        self.player2.current_state = self.board_state.copy()

        rewards = [0.0]

        for iter in range(iters):

            if show_board:
                print('RUNNING NEW GAME# = {}'.format(iter))
            self.reset()

            while self.game_state['win_state'] == 0:

                try:
                    action = self.game_state['player'].act(self.board_state)
                    self.game_state['player'].current_action = action

                except:
                    print('ya')

                # Save current board state to the active player
                self.player1.current_state = self.board_state.copy()

                self.player2.current_state = self.board_state.copy()

                r = self.update(action)

                if r != 0:
                    pass

                # now update the current player
                if self.game_state['player'] == self.player1:

                    self.game_state['player'] = self.player2
                else:

                    self.game_state['player'] = self.player1

                self.game_state['player'].update(r,self.board_state)



                if show_board:
                    print(self)

            self.player1.epsilon -= self.player1.delta

            if show_board:
                print('GAME RESULT = {}'.format(self.game_state['win_state']))

            if (iter % 10) == 0:
                # simulate 10 games vs random agent
                r = self.simulate(simulate_iters)
                rewards.append(r)

        return rewards

    def simulate(self,iters,show_board=False):
        '''

        :param iters:
        :return:
        '''
        self.player1.current_state = self.board_state.copy()

        self.player2.current_state = self.board_state.copy()

        reward = 0  # from player 1 perspective

        for _ in range(iters):

            if show_board:
                print('___________________________')
                print('Running Game #{}'.format(_))
                print('___________________________')

            self.reset()

            r = 0

            while self.game_state['win_state'] == 0:

                action = self.game_state['player'].act(self.board_state)

                r = self.update(action)

                self.player1.current_state = self.board_state.copy()

                self.player2.current_state = self.board_state.copy()

                # now update the current player
                if self.game_state['player'] == self.player1:

                    self.game_state['player'] = self.player2
                else:

                    self.game_state['player'] = self.player1

                if show_board:
                    print(self)

            # game finished
            if r == 2:
                continue
            else:
                reward += r

        reward = reward / iters

        return reward

    def update(self,action):
        '''
        Take in an action from a player
        and update the game state
        :param action: integer of action
        :return:
        '''

        p_side = self.game_state['player'].side
        if self.is_legal(action):

            board_action = list(self.board_state.keys())[action]
            # update the board
            self.board_state[board_action] = p_side

            # check if this is a winning move
            if self.is_win():
                if p_side == 'X':
                    self.game_state['win_state'] = 1

                    return 1
                else:
                    self.game_state['win_state'] = 0

                    return 0

            elif self.is_draw():
                self.game_state['win_state'] = 0.5
                return 0.5

            return 0

        else:
            print('The action {} by player {} not legal!!!'.format(p_side,action))
            return -1



if __name__ == '__main__':

    player1 = TabularQPlayer(side='X')
    player1.init()

    player2 = RandomPlayer(side='O')

    tic_tac_toe = TicTacToe(player1,player2)

    reward = tic_tac_toe.run(iters=1000,simulate_iters=10,show_board=True)

    plt.plot([x for x in range(len(reward))],reward)

    plt.show()

    player1.visualize_Q()

    tic_tac_toe.simulate(iters=15,show_board=True)

    print('Done running Tic Tac toe')


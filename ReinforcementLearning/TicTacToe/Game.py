import numpy as np





class TicTacToe:

    def __init__(self):

        self.board = '''
                | {s1} | {s2} | {s3} |    
               --------------------------
                | {s4} | {s5} | {s6} |    
               --------------------------
                | {s7} | {s8} | {s9} |  
            '''

        self.pieces = {0:'',1:'X',-1:'O'}
        self.game_state = {'s1':' ','s2':' ','s3':' '
                        ,'s4':' ','s5':' ','s6':' ',
                           's7':' ','s8':' ','s9':' '}

    def __str__(self):

        return self.board.format(** self.game_state)


    def is_legal(self,action):
        '''
        Checks to make sure an action is legally possible
        :return:
        '''
        pass


    def update(self,action):
        '''
        Take in an action from a player
        and update the game state
        :param action: tuple (player, board coordinate) i.e. ('X', s1)
        :return:
        '''

        if self.is_legal(action):
            pass
        else:
            print('The action {} by player {} not legal!!!'.format(action[0],action[1]))
            return False






if __name__ == '__main__':

    tic_tac_toe = TicTacToe()

    print(tic_tac_toe)



import random
import numpy as np


class Dealer:

    def __init__(self,decks):
        # integer representing the number of decks the dealer has
        self.decks = decks
        self.shoe = []
        self.showing_card = None
        self.current_score = None

    def deal_card(self):

        draw_card = random.choice(self.shoe)

        self.shoe.remove(draw_card)

        return draw_card

    def deal_and_replace(self):

        rand_card = np.random.randint(1,14)
        if rand_card >= 10:
            return 10
        else:
            return rand_card
    # pass in deck and return random sorted deck
    def shuffle(self,deck):

        old_deck = deck
        new_deck = []
        while(len(old_deck) > 0):

            draw = random.choice(old_deck)
            new_deck.append(draw)
            old_deck.remove(draw)

        return new_deck

    # either set the shoe for the first time or reset it
    def set_shoe(self):

        clean_shoe = []
        for di in range(self.decks):
            # dont have to worry about suites for blackjack
            for i in range(1,5):
                # A = 1
                # 2 = 2
                # ....
                for j in range(1,14):
                    clean_shoe.append(j)


        shuffle_shoe = self.shuffle(clean_shoe)
        self.shoe = shuffle_shoe

    def make_play(self,cards):

        sum_cards = np.array(cards).sum()

        if sum_cards == 2:
            return "HIT"

        if 1 in cards:
            sum_cards_use_ace = sum_cards + 10
            if sum_cards_use_ace <= 21:
                sum_cards = sum_cards_use_ace

        self.current_score = sum_cards
        if sum_cards < 17:
            return "HIT"

        elif sum_cards > 21:

            return "BUST"

        else:

            return "STAY"
#!/usr/bin/env python3

import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid

def main():

    env_name = 'MiniGrid-Empty-8x8-v0'
    # Load the gym environment
    env = gym.make(env_name)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        # Screenshot funcitonality
        elif keyName == 'ALT':
            screen_path = options.env_name + '.png'
            print('saving screenshot "{}"'.format(screen_path))
            pixmap = env.render('pixmap')
            pixmap.save(screen_path)
            return

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)

        print('step=%s, reward=%.2f' % (env.step_count, reward))
        print('direction = {}'.format(env.agent_dir))
        print('direction vector {}'.format(env.dir_vec))
        print('front vector {}'.format(env.front_pos))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
from current_config import CurrentConfig


def take_one_step(state_given, env, pertub=False):
    cur_conf = CurrentConfig()
    observation = state_given
    action, next_action, expected_state = cur_conf.uccscart.get_best_action(observation, cur_conf.meanprob)
    # if doing well pertub it so we can better chance of detecting novelties
    """
        ra = int(state_given[0] * 10000) % 4
        if (pertub and
                (ra == 0) and
                ((abs(state_given[0]) < .2)
                 and (abs(state_given[1]) < .25)
                 and (abs(state_given[2]) < .05)
                 and (abs(state_given[3]) < .1))):
            if (action == 1):
                action = 0
            else:
                action = 1
    """
    # print("Flipped Action, state=",state_given)
    cur_conf.tick = cur_conf.uccscart.tick

    return action, expected_state

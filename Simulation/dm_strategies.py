import numpy as np
import json


################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2


################################

def correct_action(information):
    if information["hotel_value"] >= 8:
        return 1
    else:
        return 0


def random_action(information):
    return np.random.randint(2)


def user_rational_action(information):
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    if information["bot_message"] >= 9:
        return 1
    else:
        return 0


def user_sloppy(information):
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_picky_short_t4t(information):
    if information["bot_message"] >= 9 or ((information["bot_message"] >= 8) and (
            len(information["previous_rounds"]) == 0 or (
            information["previous_rounds"][-1][REVIEWS].mean() >= 8))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def history_and_review_quality1(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0
    rand_p= np.random.rand(0,1)
    if rand_p<0.3:
        return correct_action
    else:
        return func

def history_and_review_quality(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0
    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"].loc[topic])*2/(rank+1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"].loc[topic])*2/(rank+1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0
    return func

def semi_corect_action(information):
    rand_p = np.random.rand(0, 1)
    if rand_p < 0.6:
        return correct_action
    else:
        user_short_t4t(information)

def LLM_based(is_stochastic):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)
        return func
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)
        return func

def LLM_based1(is_stochastic):
    rand_p= np.random.rand(0,1)
    if rand_p<0.3:
        return correct_action
    else:
        with open(f"data/baseline_proba2go.txt", 'r') as file:
            proba2go = json.load(file)
            proba2go = {int(k): v for k, v in proba2go.items()}

        if is_stochastic:
            def func(information):
                review_llm_score = proba2go[information["review_id"]]
                return int(np.random.rand() <= review_llm_score)
            return func
        else:
            def func(information):
                review_llm_score = proba2go[information["review_id"]]
                return int(review_llm_score >= 0.5)
            return func

def confidence_based_action(information):
    confidence_threshold = 0.75
    if len(information["previous_rounds"]) > 0:
        for r in information["previous_rounds"]:
            print(r)
        correct_decisions = sum(1 for r in information["previous_rounds"] if r[BOT_ACTION] == correct_action(r))
        confidence = correct_decisions / len(information["previous_rounds"])
    else:
        confidence = 1.0  # If no previous rounds, assume high confidence initially

    if confidence >= confidence_threshold:
        return user_rational_action(information)
    else:
        return random_action(information)


def cost_benefit_hybrid(information):
    # Define personal cost threshold
    cost_threshold = 7

    # Calculate perceived quality based on bot message and previous rounds
    perceived_quality = information["bot_message"]

    # Add variation based on previous rounds
    if len(information["previous_rounds"]) > 0:
        last_round = information["previous_rounds"][-1]
        if last_round[BOT_ACTION] >= 8 and last_round[REVIEWS].mean() >= 8:
            perceived_quality += 0.7  # Trust the bot more if the previous recommendation was good
        else:
            perceived_quality -= 0.4  # Trust the bot less if the previous recommendation was bad

    # Perform cost-benefit analysis
    if perceived_quality >= cost_threshold:
        return 1  # Go to the hotel
    else:
        return 0  # Don't go to the hotel

def rl_based_strategy(information):
    """
    Reinforcement Learning based strategy that updates decision based on rewards from previous rounds.
    """
    # Example parameters for the RL model
    learning_rate = 0.1
    discount_factor = 0.9

    # Initialize Q-values
    if 'q_values' not in information:
        information['q_values'] = np.zeros((10, 2))  # Assume 10 states and 2 actions (go or not go)

    state = max(int(information['bot_message'])-1 ,0) # Simplified state representation
    action = np.argmax(information['q_values'][state])  # Choose the action with the highest Q-value

    # Simulate the reward (this should be replaced with actual reward calculation)
    reward = 1 if (state >= 8 and action == 1) or (state < 8 and action == 0) else 0

    # Update Q-values based on the reward received
    next_state = state  # Assuming next state is the same for simplification
    information['q_values'][state, action] += learning_rate * (
            reward + discount_factor * np.max(information['q_values'][next_state]) - information['q_values'][state, action])

    return action

# def classifier_based_strategy(information):
#     """
#     Classifier-based strategy that predicts the action based on the input features.
#     """
#     review=information["review_features"]
#
#     return action
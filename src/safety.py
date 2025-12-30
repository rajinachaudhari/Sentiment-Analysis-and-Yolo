import yaml

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

def is_unsafe(question):
    for word in config["safety"]["blocked_words"]:
        if word in question.lower():
            return True
    return False

def safe_answer(answer):
    if len(answer.strip()) < 10:
        return "Could not generate a safe answer."
    return answer

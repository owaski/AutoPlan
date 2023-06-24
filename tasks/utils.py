import sys

PRICE = {
    'gpt-3.5-turbo': {
        'prompt': 0.002,
        'sample': 0.002,
    },
    'gpt-3.5-turbo-0301': {
        'prompt': 0.002,
        'sample': 0.002,
    },
    'gpt-3.5-turbo-0613': {
        'prompt': 0.002,
        'sample': 0.002,
    },
    'gpt-3.5-turbo-16k': {
        'prompt': 0.003,
        'sample': 0.004,
    },
    'gpt-4': {
        'prompt': 0.03,
        'sample': 0.06,
    },
    'gpt-4-0314': {
        'prompt': 0.03,
        'sample': 0.06,
    },
    'gpt-4-0613': {
        'prompt': 0.03,
        'sample': 0.06,
    },
    'text-davinci-003': {
        'prompt': 0.02,
        'sample': 0.02,
    },
    'text-davinci-002': {
        'prompt': 0.02,
        'sample': 0.02,
    }
}

def pretty_print(role, text, verbose=False):
    string = '------------{}-----------\n{}\n'.format(role, text)
    if verbose: print(string, end=''), sys.stdout.flush()
    return string
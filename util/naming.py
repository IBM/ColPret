import re


def to_str(num):
    try:
        num = float(num)
        if num > 1e11:
            return f"{num / 1e12:.2f}T"
        elif num > 1e8:
            return f"{num / 1e9:.2f}B"
        elif num > 1e5:
            return f"{num / 1e6:.2f}M"
        elif num > 1e2:
            return f"{num / 1e3:.2f}K"
        else:
            return str(num)
    except:
        return str(num)


def letter_to_scale(letter):
    if letter.lower() == "t":
        scale = 1e12
    elif letter.lower() == "b":
        scale = 1e9
    elif letter.lower() == "m":
        scale = 1e6
    elif letter.lower() == "k":
        scale = 1e3
    else:
        raise ValueError(f"Unexpected form for an int:{letter}")
    return scale


extract_letter = "([\.\d]+)([a-zA-Z])(\d*)"
extract_letter = re.compile(extract_letter)


def to_int(string):
    try:
        return int(string)
    except:
        string = string.strip()
        # letter = string[-1]
        match = re.match(extract_letter, string)
        letter = match.group(2)
        number = match.group(1)
        if match.group(3):
            number += "." + match.group(3)
        scale = letter_to_scale(letter)
        num = float(number) * scale
        return num

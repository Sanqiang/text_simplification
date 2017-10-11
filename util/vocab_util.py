from util import constant


def is_numeric(word):
    if not word:
        return False

    i = 0
    if word[0] == '-' or word[0] == '+':
        i += 1
        if i == len(word):
            return False

    while i < len(word):
        if not word[i].isnumeric():
            return False
        i += 1

    return True


def data_parse(word):
    if word == '``':
        return "\""
    elif word == '`':
        return "'"
    elif word == '\'\'':
        return "\""
    else:
        return word

if __name__ == '__main__':
    print(is_numeric('1'))
    print(is_numeric('+1'))
    print(is_numeric('-1'))
    print(is_numeric('-'))
    print(is_numeric('+'))
    print(is_numeric('-AAA-'))

    print(data_parse("''"))
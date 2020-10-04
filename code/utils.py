import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"


def sentence_tokenize(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" +
                  alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets +
                  "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)

    text = re.sub("([0-9])" + "[.]" + "([0-9])", "\\1<prd>\\2", text)

    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")

    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")

    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def transform_format(dd):

    dd = re.sub(r"Let\'s", " Let us ", dd)
    dd = re.sub(r"let\'s", " let us ", dd)
    dd = re.sub(r"\'m", " am ", dd)
    dd = re.sub(r"\'ve", " have ", dd)
    dd = re.sub(r"can\'t", " can not ", dd)
    dd = re.sub(r"n\'t", " not ", dd)
    dd = re.sub(r"\'re", " are ", dd)
    dd = re.sub(r"\'d", " would ", dd)
    dd = re.sub(r"\'ll", " will ", dd)
    dd = re.sub(r"y\'all", " you all ", dd)

    return dd


def check_ack_word(word):
    for i in range(0, len(word)):
        if ord(word[i]) < 128:
            pass
        else:
            return 0
    return 1


def standard_categorical(p):
    # batch * n_class
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False
    return prior



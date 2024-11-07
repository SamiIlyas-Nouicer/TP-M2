from collections import Counter, defaultdict


list = ["s", "t", "r", "i", "n", "g", "s",
        "t", "r", "i", "n", "g", "s", "s", "s", "M", "M", "M"]

word_freq = defaultdict(int)

term_freq = Counter(list)
print(term_freq)

print("set:", set(list))

print(word_freq)

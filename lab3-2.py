spam_dict = {"free money offer now": True, 
              "call me now": True,
              "can you call me later": False,
              "did you get the money": False,
              "let's meet later": False}

# 1. Calculate Class Priors: Calculate the prior probabilities, P(Spam) and P(Ham).
p_spam = sum(value for value in spam_dict.values())/len(spam_dict)
p_not_spam = 1 - p_spam

# Dictionary keyed on word with the tuple (not_spam_occurrences, spam_occurrences)
word_dict = {}
spam_word_count = 0

for sentence in spam_dict:
    is_spam = spam_dict[sentence]
    for word in sentence.split():
        spam_word_count += int(is_spam)
        if word in word_dict:
            word_tuple = word_dict[word]
            word_dict[word] = (word_tuple[0]+int(not(is_spam)), word_tuple[1]+int(is_spam))
        else:
            word_dict.update({word: (int(not(is_spam)), int(is_spam))})

word_count_in_spam = sum(word_tuple[1] for word_tuple in word_dict.values())
word_count_not_spam = sum(word_tuple[0] for word_tuple in word_dict.values())
vocab_size = len(word_dict)

# 2. Calculate Word Likelihoods: Calculate P(word|Class) for ”free” and ”call” using the
# add-one smoothing formula: P(word|Class) = (count(word in Class) + 1) / (total words in Class + V)

print(word_dict)

# For word "free"
p_free_spam = (word_dict["free"][1]+1) / (word_count_in_spam + vocab_size)
p_free_not_spam = (word_dict["free"][0]+1) / (word_count_not_spam + vocab_size)

# For word "call"
p_call_spam = (word_dict["call"][1]+1) / (word_count_in_spam + vocab_size)
p_call_not_spam = (word_dict["call"][0]+1) / (word_count_not_spam + vocab_size)

print(f"Message spam score is {p_call_spam * p_free_spam * p_spam:.4f}")
print(f"Message not spam score is {p_call_not_spam * p_free_not_spam * p_not_spam:.4f}")
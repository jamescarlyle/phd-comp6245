spam_dict = {"free money offer now": True, 
              "call me now": True,
              "can you call me later": False,
              "did you get the money": False,
              "let's meet later": False}

# 1. Calculate Class Priors: Calculate the prior probabilities, P(Spam) and P(Ham).
p_spam = sum(value for value in spam_dict.values())/len(spam_dict)
p_not_spam = 1 - p_spam

print(f"p_spam {p_spam}")
print(f"p_not_spam {p_not_spam}")

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
print(vocab_size)
print(word_count_in_spam)
print(word_count_not_spam)

# For word "free"
p_free_spam = (word_dict["free"][1]+1) / (word_count_in_spam + vocab_size)
p_free_not_spam = (word_dict["free"][0]+1) / (word_count_not_spam + vocab_size)


print(f"Word Free spam score is {p_free_spam:.4f}")
print(f"Word Free not spam score is {p_free_not_spam:.4f}")

# For word "call"
p_call_spam = (word_dict["call"][1]+1) / (word_count_in_spam + vocab_size)
p_call_not_spam = (word_dict["call"][0]+1) / (word_count_not_spam + vocab_size)

print(f"Word Call spam score is {p_call_spam:.4f}")
print(f"Word Call not spam score is {p_call_not_spam:.4f}")

from sklearn.datasets import fetch_20newsgroups

categories = ["sci.space", "talk.religion.misc"]
train_data = fetch_20newsgroups(subset ="train", categories = categories, shuffle = True, random_state =42)
test_data = fetch_20newsgroups(subset ="test", categories = categories, shuffle = True, random_state =42)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words ="english")
X_train = vectorizer.fit_transform(train_data.data )
X_test = vectorizer.transform(test_data.data )
y_train = train_data.target
y_test = test_data.target

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

model = MultinomialNB ()
model.fit(X_train, y_train )
y_pred = model.predict(X_test )
print(classification_report(y_test, y_pred, target_names = train_data.target_names))

# Select a document
doc_index = 10
doc_vector = X_test [ doc_index ]

# Get the model's prediction
model_prediction = model.predict(doc_vector)[0]
predicted_class_name = train_data.target_names [model_prediction]

print(" Model's Prediction : '{ predicted_class_name } '")

# --- Manually calculate the score for each class ---
log_prior_space = model.class_log_prior_ [0]
log_prior_religion = model.class_log_prior_ [1]

log_likelihoods_space = model.feature_log_prob_ [0, :]
log_likelihoods_religion = model.feature_log_prob_ [1, :]

# The dot product sums the log - likelihoods for the words present in the document.
score_space = log_prior_space + doc_vector.dot(log_likelihoods_space)
score_religion = log_prior_religion + doc_vector.dot(log_likelihoods_religion)

# Manually calculate and display results.
print(f"Score(sci.space) = {score_space[0]:.4f}")
print(f"Score(talk.religion.misc) = {score_religion[0]:.4f}")
manual_prediction = int(score_religion[0] > score_space[0])
print(f"Manual Prediction: '{train_data.target_names[manual_prediction]}'")
print(f"Match: {model_prediction == manual_prediction}")

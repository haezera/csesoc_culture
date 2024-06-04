import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

"""
Find selective vs other school system in satisfaction.
Define satisfaction >= 5; most people consider 5 to be average.
"""
data = pd.read_csv("cse_culture.csv")
satif = {
    "selective": 0,
    "other": 0
}
num_selective = 0
num_other = 0
for _, row in data.iterrows():
    if "selective" in row["SCHOOL_SYSTEM"]:
        num_selective += 1
    else:
        num_other += 1

    if "selective" in row["SCHOOL_SYSTEM"] and row["CSE_SATIF"] >= 5:
        satif["selective"] += 1
    elif row["CSE_SATIF"] >= 5:
        satif["other"] += 1

rat_sel = satif["selective"] / num_selective * 100
rat_oth = satif["other"] / num_other * 100

f1 = plt.figure(1)
plt.bar(["Selective", "Other"], [rat_sel, rat_oth], color="navy", width=0.4)
plt.xlabel("Schooling system")
plt.ylabel("% of satisfied students")
plt.title("Satisfied students categorised by schooling system ($n = 79$)")
# Data shows 6% edge to selective students; probably in range of std dev.

"""
Same as above but this time most friends in high school
"""
conn_satif = {
    "true": 0,
    "false": 0
}
num_true = 0
num_false = 0

for _, row in data.iterrows():
    if row["HS_FRIENDS"] is True:
        num_true += 1
        if row["CSE_SATIF"] >= 5:
            conn_satif["true"] += 1
    else:
        num_false += 1
        if row["CSE_SATIF"] >= 5:
            conn_satif["false"] += 1

conn_t_ratio = conn_satif["true"] / num_true * 100
conn_f_ratio = conn_satif["false"] / num_false * 100

f2 = plt.figure(2)
plt.bar(["True", "False"], [conn_t_ratio, conn_f_ratio], color="maroon", width=0.4)
plt.xlabel("Most friends come from high school/high school connections")
plt.ylabel("% of satisfied students")
plt.title("Satisfied students categorised by friendship origins ($n = 79$)")
# Data shows the inverse of what we expect from figure 1 - having friendships
# carry over from high school have a negative effect.

"""
CSE Inclusion for school systems
"""
s_inclusion = {
    "selective": 0,
    "other": 0
}

for _, row in data.iterrows():
    if "selective" in row["SCHOOL_SYSTEM"] and row["CSE_INCLUSION"] >= 5:
        s_inclusion["selective"] += 1
    elif row["CSE_INCLUSION"] >= 5:
        s_inclusion["other"] += 1

s_inc_ratio = s_inclusion["selective"] / num_selective * 100
o_inc_ratio = s_inclusion["other"] / num_other * 100

f3 = plt.figure(3)
plt.bar(["Selective", "Other"], [s_inc_ratio, o_inc_ratio], width=0.4)
plt.xlabel("Schooling system")
plt.ylabel("% of students who feel included")
plt.title("Student inclusion categorised by schooling system ($n = 79$)")
# Similar to friendship for selective

"""
Sentiment analysis for notes.
"""
s_sentiment = {
    "selective": 0,
    "other": 0
}
s_responded = 0
o_responded = 0
for _, row in data.iterrows():
    if row["NOTES"] is not None:
        blob = TextBlob(str(row["NOTES"]))
        if "selective" in row["SCHOOL_SYSTEM"]:
            s_responded += 1
            s_sentiment["selective"] += blob.sentiment.polarity * (1 - blob.sentiment.subjectivity)
        else:
            o_responded += 1
            s_sentiment["other"] += blob.sentiment.polarity * (1 - blob.sentiment.subjectivity)

f4 = plt.figure(4)
plt.bar(["Selective", "Other"], [s_sentiment["selective"] / s_responded, s_sentiment["other"] / o_responded])
plt.xlabel("Schooling system")
plt.ylabel("Subjectivity-weighted polarity of statements")
plt.title("Subjectivity-weighted polarity of statements given by students, categorised by schooling system")

plt.show()



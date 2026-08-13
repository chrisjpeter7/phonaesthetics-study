import pandas as pd
import numpy as np

from scipy.stats import ttest_rel
from scipy.stats import ttest_ind

df = pd.read_csv('raw_data.csv')

print(df.columns.tolist())
print(df.shape)

df = df.rename(columns = {
    'Participant #': 'participant',
    'Gender': 'gender',
    'Word': 'word',
    'Consonant':'c_type',
    'Vowel':'v_type',
    'Rating':'rating'
})

df['C1'] = df['word'].str[0]
df['V'] = df['word'].str[1]
df['C2'] = df['word'].str[2]

meansvowel = (
    df.groupby(
        ['participant', 'v_type']
    )['rating']
    .mean()
    .unstack()
)
print("\nVowel Analysis")

meansvowel["fbdiff"] = meansvowel["Front"] - meansvowel["Back"]
print("Mean Front:", meansvowel["Front"].mean())
print("Mean Back:", meansvowel["Back"].mean())
print("Mean difference:", meansvowel["fbdiff"].mean())

#In the original study, I compared the FB score to 0
vowelp = ttest_rel(
    meansvowel["Front"],
    meansvowel["Back"]
)
print("P-Value: ", vowelp.pvalue)


meanscons = (
    df.groupby(
        ['participant', 'c_type']
    )['rating']
    .mean()
    .unstack()
)
print("\nConsonant Analysis")

meanscons["psdiff"] = meanscons["Plosive"] - meanscons["Sonorant"]
print("Mean Plosive:", meanscons["Plosive"].mean())
print("Mean Sonorant:", meanscons["Sonorant"].mean())
print("Mean difference:", meanscons["psdiff"].mean())

#In the original study, I compared the PS score to 0
consp = ttest_rel(
    meanscons["Plosive"],
    meanscons["Sonorant"]
)
print("P-Value: ", consp.pvalue)


print("\nCombined")
combined = (
    df.groupby(["c_type", "v_type"])["rating"]
      .mean()
)

print(combined)

print("\nGender Analysis of PS Score (FB Score Differences were Negligible)")
gender = df.groupby("participant")["gender"].first()

meansvowel = meansvowel.join(gender)
meanscons = meanscons.join(gender)

print(
    meanscons.groupby("gender")["psdiff"].agg(
        ["mean", "std", "count"]
    )
)

men = meanscons.loc[
    meanscons["gender"] == "M",
    "psdiff"
]
women = meanscons.loc[
    meanscons["gender"] == "F",
    "psdiff"
]
welch = ttest_ind(
    men,
    women,
    equal_var=False
)
print("P-Value: ",welch.pvalue)

print("\nIndividual Consonant 1 Means")
cons1 = (
    df.groupby(["C1"])['rating'].agg(["mean", "count"])
)
print(cons1)

print("\nIndividual Vowel 1 Means")
vow1 = (
    df.groupby(["V"])['rating'].agg(["mean", "count"])
)
print(vow1)

print("\nIndividual Consonant 2 Means")
cons2 = (
    df.groupby(["C2"])['rating'].agg(["mean", "count"])
)
print(cons2)
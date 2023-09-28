# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 📦 Requirements & login

# !gcloud auth application-default login

# !pip install google-cloud-translate==2.0.1
# !pip install --upgrade deepl

import difflib

# + pycharm={"is_executing": true}
import pandas as pd

# -

# # 💾 Load original mimic content

df = pd.read_csv("data/NOTEEVENTS.csv")
original_text = df.TEXT

# # 💾 Load Rian's Translation

with open("data/rian_translation.txt", "r") as file:
    rian_text = [line for line in file.readlines() if line != "\n"]


def print_side_by_side(a, b, size=100, space=10):
    while a or b:
        print(a[:size].ljust(size) + " " * space + b[:size])
        a = a[size:]
        b = b[size:]


TEXT_INDEX = 2

# # 😵 Original text vs. Rian's Translation

print_side_by_side(
    original_text[TEXT_INDEX].replace("\n", " ").replace("\r", " ").strip(), rian_text[TEXT_INDEX]
)


# # 🥶 Rian's Translation vs. Google Translation


def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    return result["translatedText"]


diff = difflib.ndiff(
    rian_text[TEXT_INDEX].split(".")[:50],
    [translate_text("fr", line) for line in original_text[TEXT_INDEX].split(".")[:50]],
)
print("\n".join(list(diff)))


# # 🫠 Rian's Translation vs. DeepL Translation


def deepl_translation(target, text):
    import deepl

    auth_key = "272f77e7-34a0-d1be-8847-ffa31e1ccb2e"  # Replace with your key
    translator = deepl.Translator(auth_key)

    result = translator.translate_text(
        text, target_lang=target, split_sentences="off", preserve_formatting=True
    )
    return result.text


diff = difflib.ndiff(
    rian_text[TEXT_INDEX].split(".")[:50],
    [deepl_translation("fr", line) for line in original_text[TEXT_INDEX].split(".")[:50]],
)
print("\n".join(list(diff)))

# # 🪱 Google Translation vs. DeepL Translation

# + pycharm={"is_executing": true}
diff = difflib.ndiff(
    [translate_text("fr", line) for line in original_text[TEXT_INDEX].split(".")[:50]],
    [deepl_translation("fr", line) for line in original_text[TEXT_INDEX].split(".")[:50]],
)
print("\n".join(list(diff)))
# -

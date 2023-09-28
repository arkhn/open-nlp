# 🪱 MIMIC-III Translation & Pseudonymisation

This repo uses Google Translate API to translate MIMIC-III to French.
Then, we use Faker to pseudonymised a set of MIMIC-III.

## 🎲 Instructions

### 🐊 Translation

1. [generate and download](https://console.cloud.google.com/iam-admin/serviceaccounts/details/109771817206229132137/keys?project=leafy-ember-294410)
   the json authentication key to use Google Translate API and set the environment variable
   `GOOGLE_APPLICATION_CREDENTIALS` as an absolute path to the authentification key
2. install dependencies using `poetry`
3. get the `NOTEEVENTS.csv` from Mimic-III or the `shuffled version` using `clearml`:
   ```bash
   clearml-data get --id 12f79ee456714d24bb350a9fef04769f
   ```
   and copy the content of the downloaded to the data folder 😉
4. use the following command to get the translation :

☠️ **it is important to keep the same shuffled version each time you translate a new slice of mimic. Otherwise, you risk to get some overlap between
the document you'll translate, the translation is soooo expensive !**

```bash
# get a shuffled version of Mimic-III (a new shuffling but you don't need this if you already have the shuffled version of Mimic-III)
python mimic_translation/translate.py shuffle data/raw/NOTEEVENTS.csv ./data/raw/mimic-III-shuffled.csv
# extract the text from csv
python mimic_translation/translate.py extract-mimic-texts ./data/raw/mimic-III-shuffled.csv \
                                                          ./data/translation/en \
                                                          0 1100
# extract all substitutions tokens from Mimic-III to create a glossary for Google Translate API
./scripts/create_glossary.sh  "./data/translation/en/0_1100/*" > ./data/translation/glossary_0_1100.csv
./scripts/upload_glossary_to_gcp.sh  ./data/translation/glossary_0_1100.csv mimic-iii-tokens-1100
# see the price or your translation
python mimic_translation/translate.py pricing ./data/translation/en/0_1100
# translate with Google Translate API
python mimic_translation/translate.py translate ./data/translation/en/0_1100 ./data/translation/fr --glossary-id mimic-iii-tokens-1100
```

### 🃏 Substitution

1. create your regexps in a text file such as `data/subsitution/your_re-rules.txt`. Each regex should represent a group of entities
2. extract the whole anonymised tokens of the sub-corpus using
   ```bash
   egrep -Eo '\[\*.*?\*.*?\*.*?\* ?]' data/translation/en/0_1100/* | sed 's/.*://g' | sort | uniq > data/substitution/en_mimic_iii_tokens_0_1100.txt
   ```
3. verify if you cover all the tokens using:
   ```bash
   python python mimic_translation/substitution.py test-boring-regex-rules ./data/substitution/re-rules.txt ./data/substitution/en_mimic_iii_tokens_0_1100.txt
   ```
4. add the missing regexps into the `generate_fake_entity` methods with the associated substitutions
5. substitute the whole pseudo-labels with the following command:
   ```bash
   python substitution.py de-anonymised ./data/substitution/re-rules.txt ./data/translation/fr/0_1100 ./data/fr_substitution/0_1100
   ```
6. to upload a new version of the dataset in clearml:
   ```bash
   clearml-data create --name mimic-translation --project pseudonimysation
   clearml-data add --files data
   clearml-data close --id $(id)
   ```

### 💝 Extra (more fun !)

You can get the price of a translation using `python pricing --input-path ./txt`

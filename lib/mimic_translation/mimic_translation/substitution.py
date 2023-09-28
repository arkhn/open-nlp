import json
import os
import pathlib
import random
import re
from functools import partial
from random import choice

import pandas as pd
import typer
from faker import Faker

app = typer.Typer()
# Open the database with all hospitals names in France.
with open("./data/substitution/les_etablissements_hospitaliers_franciliens.json") as hospitals:
    hospital_json = json.load(hospitals)
    HOSPITALS_LIST = [hospital["fields"]["raison_sociale"].title() for hospital in hospital_json]


@app.command()
def check_boring_regex_rules(
    regex_path: str,
    extracted_labels_path: str,
):
    """Test if all regex rules in the regex files remove all the anonymised labels

    Args:
        regex_path : Path to the file containing the regex rules.
        extracted_labels_path : Path to the file containing the extracted labels.
    """
    with open(extracted_labels_path, "r") as labels_file:
        labels = "".join(labels_file.readlines())
        len_labels = len(labels.split("\n"))
        match_count = 0
        with open(regex_path, "r") as regex_rules:
            for regex_rule in regex_rules:
                match_count += len(re.findall(regex_rule[:-1], labels))
        if match_count != len_labels - 1:
            labels_sub = labels
            with open(regex_path, "r") as regex_rules:
                for regex_rule in regex_rules:
                    labels_sub = re.sub(regex_rule[:-1], "", labels_sub)
            sub = re.sub(r"\n{2}", "", labels_sub)
            raise ValueError(f"Sorry 🥳 here are the remained tokens:\n\n {sub}")
        else:
            print("matching tests are passed 😏")


@app.command()
def de_anonymised(
    regex_path: str,
    mimic_text_folder_path: str,
    output_path: str,
):
    """De-anonymises the corpus using fake entities.
    De-anonymises a set of text files in a given folder by replacing the anonymised labels with fake
    entities generated using a given regex file.

        Args:
            regex_path (str): Path to the file containing the regex rules.
            mimic_text_folder_path (str): Path to the folder containing the text files
                to be de-anonymised.
            output_path (str): Path to the output folder where the de-anonymised text
                files will be saved.
    """
    fake = Faker(["fr-FR"])
    seed = 42
    Faker.seed(seed)
    random.seed(seed)
    pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)

    regexps_dict = {
        r"\[\*\*[0-9]+-/[0-9]+\*\*\]": partial(fake.date, pattern="%m-%Y"),
        r"\[\*\*[0-9]+/[0-9]+\*\*\]": partial(fake.date, pattern="%m/%Y"),
        r"\[\*\*-[0-9]+/[0-9]+\*\*\]": partial(fake.date, pattern="%m/%Y"),
        r"\[\*\*[0-9]+-[0-9]+-\*\*\]": partial(fake.date, pattern="%m-%Y"),
        r"\[\*\*[0-9]+-[0-9]+\*\*]": partial(fake.date, pattern="%m-%Y"),
        r"\[\*\*[0-9]+\*\*]": partial(random.randint, 1, 12),
        r"\[\*\* [0-9]+\*\*]": "[** **]",
        r"\[\*\*[0-9]+-[0-9]+-[0-9]+\*\*]": partial(fake.date, pattern="%d-%m-%Y"),
        r"\[\*\*Hospital[0-9]* [0-9]*\*\*]": partial(get_hospital),
        r"\[\*\*Name[0-9]* \(MD\) [0-9]*\*\*]": partial(get_doctor_name, fake),
        r"\[\*\*Name[0-9]* \(NI\) [0-9]*\*\*]": partial(fake.first_name),
        r"\[\*\*Name[0-9]* \(PRE\) [0-9]*\*\*]": partial(get_professor_name, fake),
        r"\[\*\*Name[0-9]* \(PTitle\) [0-9]*\*\*]": partial(get_professor_name, fake),
        r"\[\*\*Name[0-9]* \(STitle\) [0-9]*\*\*]": partial(fake.last_name),
        r"\[\*\*Name[0-9]* \(NameIs\) [0-9]*\*\*]": partial(fake.last_name),
        r"\[\*\* \*\*\]": "[** **]",
        r"\[\*\*CC Contact Info .*?\*\*\]": partial(fake.email),
        r"\[\*\*E-mail address.*?\*\*\]": partial(fake.email),
        r"\[\*\*Name Prefix.*?\*\*\]": partial(get_prefix_name, fake),
        r"\[\*\*Name Initial \(MD\).*?\*\*\]": partial(get_name_initial, fake),
        r"\[\*\*Name Initial \(PRE\).*?\*\*\]": partial(get_name_initial, fake),
        r"\[\*\*Name Initial \(NameIs\).*?\*\*\]": partial(get_name_initial, fake),
        r"\[\*\*Year \(4 digits\).*?\*\*\]": partial(fake.year),
        r"\[\*\*Medical Record.*?\*\*\]": partial(fake.random_number, fix_len=True, digits=7),
        r"\[\*\*Pager number.*?\*\*\]": partial(fake.phone_number),
        r"\[\*\*Street Address.*?\*\*\]": partial(get_address, fake),
        r"\[\*\*Age over 90.*?\*\*\]": partial(random.randint, 90, 100),
        r"\[\*\*Initial.*?\*\*\]": partial(get_name_initial, fake),
        r"\[\*\*Apartment Address.*?\*\*\]": partial(get_address, fake),
        r"\[\*\*Unit Number.*?\*\*\]": partial(fake.random_number, fix_len=True, digits=3),
        r"\[\*\*Wardname.*?\*\*\]": partial(get_ward_name),
        r"\[\*\*Telephone/Fax.*?\*\*\]": partial(fake.phone_number),
        r"\[\*\*Numeric Identifier.*?\*\*\]": partial(fake.random_number, fix_len=True, digits=6),
        r"\[\*\*Last Name.*?\*\*\]": partial(fake.last_name),
        r"\[\*\*Known lastname.*?\*\*\]": partial(fake.last_name),
        r"\[\*\*Month \(only\).*?\*\*\]": partial(fake.month_name),
        r"\[\*\*Location.*?\*\*\]": partial(fake.city),
        r"\[\*\*Known firstname.*?\*\*\]": partial(fake.first_name),
        r"\[\*\*MD Number.*?\*\*\]": partial(fake.random_number, fix_len=True, digits=4),
        r"\[\*\*Hospital Ward.*?\*\*\]": partial(get_ward_name),
        r"\[\*\*Hospital Unit.*?\*\*\]": partial(get_hospital),
        r"\[\*\*First Name.*?\*\*\]": partial(fake.first_name),
        r"\[\*\*Doctor First.*?\*\*\]": partial(fake.first_name),
        r"\[\*\*Doctor Last.*?\*\*\]": partial(fake.last_name),
        r"\[\*\*Clip Number \(Radiology\).*?\*\*\]": partial(
            fake.random_number, fix_len=True, digits=5
        ),
        r"\[\*\*Date range.*?\*\*\]": partial(get_duration),
        r"\[\*\*Date Range.*?\*\*\]": partial(get_duration),
        r"\[\*\*Country.*?\*\*\]": partial(fake.country),
        r"\[\*\*Month/Day/Year.*?\*\*\]": partial(fake.date, pattern="%d/%m/%Y"),
        r"\[\*\*Month/Day .*?\*\*\]": partial(fake.date, pattern="%d/%m"),
        r"\[\*\*Month/Year .*?\*\*\]": partial(fake.date, pattern="%m/%Y"),
        r"\[\*\*State .*?\*\*\]": partial(fake.administrative_unit),
        r"\[\*\*Male First Name .*?\*\*\]": partial(fake.first_name_male),
        r"\[\*\*Job Number .*?\*\*\]": partial(fake.random_number, fix_len=True, digits=5),
        r"\[\*\*Female First Name .*?\*\*\]": partial(fake.first_name_female),
    }
    entities: dict = {
        "document_id": [],
        "labels": [],
        "offsets": [],
    }
    for file in os.listdir(mimic_text_folder_path):
        with open(os.path.join(mimic_text_folder_path, file), "r") as txt:
            doc = txt.read()
            matches_offset = []
            labels = []
            offset_shifting = 0
            matches: list = []
            pseudo_entities = {}
            for regex in regexps_dict.keys():
                if re.search(regex, doc) is not None:
                    regex_matches = list(re.finditer(regex, doc))
                    matches += regex_matches
                    matches_set = regex_matches
                    pseudo_entities.update(
                        {
                            pseudo_entity.group(): generate_fake_entity(regex, fake, regexps_dict)
                            for pseudo_entity in matches_set
                        }
                    )
            for match in sorted(matches, key=lambda x: x.start()):
                replaced_entities = pseudo_entities[match.group()]
                doc = (
                    doc[: match.start() + offset_shifting]
                    + replaced_entities
                    + doc[match.end() + offset_shifting :]
                )
                matches_offset.append(
                    (
                        match.start() + offset_shifting,
                        match.start() + offset_shifting + len(replaced_entities),
                    )
                )
                offset_shifting += len(replaced_entities) - len(match.group())
                labels.append(match.group())
            entities["document_id"].append(file)
            entities["labels"].append(labels)
            entities["offsets"].append(matches_offset)

        with open(os.path.join(output_path, file), "w") as output_file:
            output_file.write(doc)
        pd.DataFrame(entities).to_csv(os.path.join(output_path, "entities.csv"), index=False)


def get_ward_name():
    """Returns a random name of a ward in a hospital.

    Returns:
        Name of a ward in a hospital.
    """
    return choice(  # nosec
        [
            "Accueil des urgences",
            "Accueil des urgences adultes – UHCD",
            "Accueil des urgences enfants",
            "Anesthésie",
            "Bloc opératoire",
            "Cardiologie et médecine vasculaire",
            "CeGIDD",
            "Centre de référence de la maladie de Lyme",
            "Centre de vaccination",
            "Centre Médico-Psychologique (CMP)",
            "Chirurgie ambulatoire",
            "Chirurgie maxillo-faciale et stomatologie",
            "Chirurgie orthopédique et traumatologique",
            "Chirurgie pédiatrique",
            "Chirurgie viscérale et urologique",
            "Consultation de médecine générale",
            "Consultation de psychiatrie – Unité PLUCE",
            "Consultation mémoire",
            "Diabétologie – Endocrinologie",
            "Douleur et accompagnement en soins palliatifs – EMASP",
            "EHPAD/USLD « Les Vignes »",
            "Equipe mobile gériatrie et plaies cicatrisation",
            "Gynécologie Obstétrique",
            "Hépato-gastroentérologie",
            "Laboratoire de Biologie Médicale",
            "Maladies infectieuses et tropicales",
            "Médecine du sport",
            "Médecine Intensive Réanimation",
            "Neurologie",
            "Oncologie médicale",
            "Ophtalmologie",
            "ORL et chirurgie de la face et du cou",
            "Pédiatrie et néonatologie",
            "Pharmacie – stérilisation",
            "Pneumologie",
            "Radiologie et Imagerie médicale",
            "Rééducation Fonctionnelle – Kinésithérapie – Ostéopathie – Pédicurie",
            "Rhumatologie",
            "Service d’information médicale",
            "Service de Traitement des Maladies Addictives",
            "Service diététique",
            "SMUR",
            "SSR – Soins de suite et de réadaptation",
            "Sur le territoire",
            "Unité d’Hospitalisation Temps Plein (UHTP)",
            "Unité d’aval des urgences – UAU",
            "Unité de gériatrie aiguë",
            "Unité de Médecine Polyvalente",
        ]
    )


def get_hospital() -> str:
    """Get a hospital name among the HOSPITALS_LIST choosing randomly.
    Returns:
        A hospital name.
    """
    return choice(HOSPITALS_LIST)  # nosec


def get_doctor_name(fake: Faker) -> str:
    """
    Returns a fake doctor's name.

    Args:
        fake: Instance of the `Faker` class to generate fake names.

    Returns:
        Fake doctor's name.
    """
    return f"Dr. {fake.last_name()}"


def get_professor_name(fake: Faker) -> str:
    """Returns a fake professor's name.

    Args:
        fake: Instance of the `Faker` class to generate fake names.

    Returns:
        Fake professor's name.
    """
    return f"Pr. {fake.last_name()}"


def generate_fake_entity(regex: str, fake: Faker, regexps_dict: dict) -> str:
    """Generates a fake entity based on the given regex pattern.

    Args:
        regexps_dict:
        regex: Regex pattern to use to generate the fake entity.
        fake: Instance of the `Faker` class to generate fake entities.

    Returns:
        Fake entity based on the given regex pattern.
    """
    regex_faker = regexps_dict[regex]
    return str(regex_faker()) if isinstance(regex_faker, partial) else str(regex_faker)


def get_duration() -> str:
    """Get a duration among the duration list choosing randomly.
    Returns a random duration as a string, in the format "{n} {days/weeks/months}"

    Returns:
        A string representing a duration, in the format "{n} {days/weeks/months}".
    """
    return f'{random.randint(2, 4)} {choice(["jours", "semaines", "mois"])}'  # nosec


def get_prefix_name(fake: Faker) -> str:
    """Generates a random prefix name.
    Returns a random prefix for a name (e.g. "Madame", "Monsieur") followed
    by a randomly generated last name.

    Args:
        fake: A Faker object used to generate the last name.

    Returns:
        A string containing a prefix for a name followed by a last name.
    """
    return f'{choice(["Madame", "Monsieur", "Mr", "Mme"])} {fake.last_name()}'  # nosec


def get_address(fake: Faker) -> str:

    """Generates a random address.
    Returns a randomly generated address, with line breaks replaced with spaces and commas removed.

    Args:
        fake: A Faker object used to generate the address.

    Returns:
        A string containing a randomly generated address.
    """
    return fake.address().replace("\n", " ").replace(",", "")


def get_name_initial(fake: Faker) -> str:
    """Returns the initial of a randomly generated name.

    Args:
        fake: A Faker object used to generate the name.

    Returns:
        A string containing the initial of a randomly generated name.
    """
    return "".join([char for char in fake.name() if char.isupper()])


if __name__ == "__main__":
    app()

"""
This file contains the logic for loading data for all typing tasks.
# TODO license 
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import *

from transformers.tokenization_utils import SPECIAL_TOKENS_MAP_FILE

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor



class OntoNoteProcessor(DataProcessor):
    """
    `Few-NERD <https://ningding97.github.io/fewnerd/>`_ a large-scale, fine-grained manually annotated named entity recognition dataset
    It was released together with `Few-NERD: Not Only a Few-shot NER Dataset (Ning Ding et al. 2021) <https://arxiv.org/pdf/2105.07464.pdf>`_
    
    Examples:
    ..  code-block:: python
        from openprompt.data_utils.typing_dataset import PROCESSORS
        base_path = "datasets/Typing"
        dataset_name = "FewNERD"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)
        assert processor.get_num_labels() == 66
        assert processor.get_labels() == [
            "person-actor", "person-director", "person-artist/author", "person-athlete", "person-politician", "person-scholar", "person-soldier", "person-other",
            "organization-showorganization", "organization-religion", "organization-company", "organization-sportsteam", "organization-education", "organization-government/governmentagency", "organization-media/newspaper", "organization-politicalparty", "organization-sportsleague", "organization-other",
            "location-GPE", "location-road/railway/highway/transit", "location-bodiesofwater", "location-park", "location-mountain", "location-island", "location-other",
            "product-software", "product-food", "product-game", "product-ship", "product-train", "product-airplane", "product-car", "product-weapon", "product-other",
            "building-theater", "building-sportsfacility", "building-airport", "building-hospital", "building-library", "building-hotel", "building-restaurant", "building-other",
            "event-sportsevent", "event-attack/battle/war/militaryconflict", "event-disaster", "event-election", "event-protest", "event-other",
            "art-music", "art-writtenart", "art-film", "art-painting", "art-broadcastprogram", "art-other",
            "other-biologything", "other-chemicalthing", "other-livingthing", "other-astronomything", "other-god", "other-law", "other-award", "other-disease", "other-medical", "other-language", "other-currency", "other-educationaldegree",
        ]
        assert dev_dataset[0].text_a == "The final stage in the development of the Skyfox was the production of a model with tricycle landing gear to better cater for the pilot training market ."
        assert dev_dataset[0].meta["entity"] == "Skyfox"
        assert dev_dataset[0].label == 30
    """
    def __init__(self):
        super().__init__()
        self.labels = ["/other/health/malady", "/organization/company", "/other/award", "/person/legal", "/location", "/location/geograpy/island", "/other/art/broadcast", "/person/religious_leader", "/location/structure/hospital", "/location/transit", "/person/military", "/person/artist/actor", "/person/coach", "/other/language", "/person/artist/director", "/location/country", "/organization/company/news", "/other/product/computer", "/other/product/software", "/organization", "/person/doctor", "/other/art/film", "/organization/political_party", "/other/art", "/other/supernatural", "/other/language/programming_language", "/location/structure/government", "/other", "/other/living_thing/animal", "/organization/education", "/other/art/stage", "/other/event/election", "/other/health/treatment", "/location/structure/theater", "/organization/sports_league", "/other/food", "/location/city", "/organization/sports_team", "/location/structure/sports_facility", "/other/art/writing", "/location/structure/airport", "/organization/company/broadcast", "/other/internet", "/other/art/music", "/location/geography/island", "/other/scientific", "/organization/stock_exchange", "/other/product/weapon", "/person/title", "/organization/government", "/organization/music", "/organization/military", "/other/heritage", "/other/product", "/person/political_figure", "/other/legal", "/other/living_thing", "/location/celestial", "/location/geography", "/location/transit/bridge", "/person", "/other/event/protest", "/person/artist/author", "/other/health", "/location/structure", "/other/currency", "/other/event/violent_conflict", "/other/event/natural_disaster", "/location/transit/road", "/person/artist/music", "/location/transit/railway", "/other/event/holiday", "/location/structure/hotel", "/other/religion", "/person/athlete", "/person/artist", "/location/park", "/location/geography/body_of_water", "/other/event", "/other/sports_and_leisure", "/other/event/sports_event", "/other/product/car", "/location/geography/mountain", "/organization/transit", "/other/body_part", "/other/product/mobile_phone"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.txt".format(split))
        examples = []
        with open(path, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            linelist = line.strip().split('\t')
            start = int(linelist[0])
            end = int(linelist[1])
            tag = linelist[3]
            text_a = linelist[2]
            meta = {"entity": " ".join(text_a.split(" ")[start:end])}
            example = InputExample(guid=str(idx), text_a=text_a, meta=meta, label=self.get_label_id(tag))
            examples.append(example)
        return examples

class FewNerdProcessor(OntoNoteProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["organization-government/governmentagency", "organization-media/newspaper", "other-astronomything", "product-car", "event-attack/battle/war/militaryconflict", "product-train", "organization-other", "other-language", "product-airplane", "other-disease", "organization-sportsteam", "building-other", "product-weapon", "art-music", "location-mountain", "organization-education", "person-politician", "product-software", "person-actor", "other-currency", "building-hotel", "other-award", "event-protest", "location-GPE", "product-ship", "organization-sportsleague", "product-game", "building-library", "building-theater", "location-park", "person-scholar", "product-other", "other-law", "organization-showorganization", "organization-religion", "location-road/railway/highway/transit", "event-other", "other-livingthing", "building-sportsfacility", "other-chemicalthing", "art-other", "event-disaster", "person-soldier", "location-island", "event-sportsevent", "other-educationaldegree", "art-painting", "building-restaurant", "other-god", "person-director", "location-other", "building-airport", "person-athlete", "person-artist/author", "other-medical", "organization-company", "building-hospital", "location-bodiesofwater", "product-food", "event-election", "other-biologything", "art-writtenart", "art-broadcastprogram", "organization-politicalparty", "art-film", "person-other"]

class BBNProcessor(OntoNoteProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["/PRODUCT/WEAPON", "/GPE/STATE_PROVINCE", "/FACILITY", "/ORGANIZATION/POLITICAL", "/WORK_OF_ART/SONG", "/PRODUCT", "/FACILITY/BUILDING", "/LOCATION/LAKE_SEA_OCEAN", "/SUBSTANCE/CHEMICAL", "/EVENT", "/WORK_OF_ART", "/SUBSTANCE/DRUG", "/GPE/CITY", "/LOCATION/RIVER", "/ORGANIZATION/HOTEL", "/ORGANIZATION/CORPORATION", "/WORK_OF_ART/PLAY", "/WORK_OF_ART/BOOK", "/ORGANIZATION/GOVERNMENT", "/FACILITY/BRIDGE", "/ORGANIZATION/HOSPITAL", "/EVENT/WAR", "/PLANT", "/ORGANIZATION/MUSEUM", "/ANIMAL", "/FACILITY/ATTRACTION", "/SUBSTANCE/FOOD", "/LOCATION/CONTINENT", "/GPE", "/PERSON", "/LAW", "/LOCATION", "/ORGANIZATION", "/LANGUAGE", "/FACILITY/HIGHWAY_STREET", "/CONTACT_INFO/url", "/DISEASE", "/EVENT/HURRICANE", "/PRODUCT/VEHICLE", "/GPE/COUNTRY", "/SUBSTANCE", "/ORGANIZATION/EDUCATIONAL", "/GAME", "/LOCATION/REGION", "/ORGANIZATION/RELIGIOUS", "/FACILITY/AIRPORT"]

class OpenEntityProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = json.load(open("./openprompt_util/script/openentity/labels.json"))
    
    def _get_tags(self, d):
        tags = []
        for tag in d["y_str"]:
            tag_list = tag.split("_")
            tags += tag_list
        tags = list(set(tags))
        tags = [t for t in tags if t in self.labels] # map to label words
        return tags
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.json".format(split))
        examples = []
        with open(path, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            d = json.loads(line)
            tag = self._get_tags(d)
            labels = [self.get_label_id(t) for t in tag]
            entity = d["mention_span"]
            text_a = " ".join(d["left_context_token"] + [entity] + d["right_context_token"])
            meta = {"entity": entity, "label": labels}
            example = InputExample(guid=str(idx), text_a=text_a, meta=meta, label=labels)
            examples.append(example)
        return examples
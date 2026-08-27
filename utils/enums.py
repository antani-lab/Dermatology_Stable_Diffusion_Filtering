from enum import Enum


class ALG(Enum):
    mm = 0
    sd = 1
    mm_sd = 2
    mm_sd_limited = 3
    mm_sd_limited_triple = 4


class NETWORK(Enum):
    resnet18 = 0
    densenet121 = 1
    mobilenet_v2 = 2
    HIPT = 3
    ViT = 4
    ViT2 = 5
    PanDerm = 6


class PHASE(Enum):
    train = 0
    valid = 1
    test = 2
    all = 3


class MOD(Enum):
    img = 0
    txt = 1
    multimodal = 2


class COMPONENTS(Enum):
    images = 0
    reports = 1


class REPORTS(Enum):
    abcd = 0
    short = 1
    char = 2
    doc = 3
    meta = 4
    all = 5
    random = 6
    images = 7
    skingpt4_abcd = 13
    skingpt4_char = 14
    skingpt4_doc = 15
    skingpt4_p1 = 16
    skingpt4_p2 = 17
    skingpt4_meta = 18
    skingpt4_all = 19
    skingpt4_p1_all = 20
    dermlip_abcd = 21
    dermlip_char = 22
    dermlip_doc = 23
    dermlip_p1 = 24
    dermlip_p2 = 25
    dermlip_meta = 26
    dermlip_all = 27
    dermlip_p1_all = 28
    medgemma_abcd = 29
    medgemma_char = 30
    medgemma_doc = 31
    medgemma_meta = 32
    medgemma_all = 33
    whole = 34
    whole_all = 35


class CONCEPTS(Enum):
    classes = 0
    subclasses = 1
    classes_matching = 2
    subclasses_matching = 3


class PARTITION(Enum):
    internal = 0
    external = 1
    dermoscopic = 2
    clinical = 3
    whole = 4

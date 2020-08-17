from utils.citr_dut_dataset import CitrDataset, DutDataset
from utils.ind_dataset import IndDataset


class DatasetTag:
    citr = 0
    dut = 1
    ind = 2


DATASET_TAG2INFO = {
    DatasetTag.citr: CitrDataset,
    DatasetTag.dut: DutDataset,
    DatasetTag.ind: IndDataset,
}


class DataProcessor:
    """sequence classification을 위해 data를 처리하는 기본 processor"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        tensor dict에서 example을 가져오는 메소드
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """train data에서 InputExample 클래스를 가지고 있는 것들을 모으는 메소드"""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """dev data(validation data)에서 InputExample 클래스를 가지고 있는 것들을 모으는 메소드"""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """test data에서 InputExample 클래스를 가지고 있는 것들을 모으는 메소드"""
        raise NotImplementedError()

    def get_labels(self):
        """data set에 사용되는 라벨들을 리턴하는 메소드"""
        raise NotImplementedError()

    def tfds_map(self, example):
        """
        tfds(tensorflow-datasets)에서 불러온 데이터를 DataProcessor에 알맞게 가공해주는 메소드
        """
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """tab으로 구분된 .tsv파일을 읽어들이는 클래스 메소드"""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
# -*- coding: utf-8 -*-
import numpy as np
import operator
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.initializers import Constant
# from keras.preprocessing.sequence import pad_sequences


class Helpers:
    @staticmethod
    def get_max_vector_len(tweets):
        return len(max(tweets, key=len))

    @staticmethod
    def check_embeddings_coverage(word_counts, embeddings):
        covered = {}
        missing = {}
        n_covered = 0
        n_missing = 0

        for word in word_counts:
            try:
                covered[word] = embeddings[word]
                n_covered += word_counts[word]
            except KeyError:
                missing[word] = word_counts[word]
                n_missing += word_counts[word]

        vocab_coverage = round(len(covered) / len(word_counts), 5)
        text_coverage = round((n_covered / (n_covered + n_missing)), 5)
        sorted_missing = sorted(missing.items(), key=operator.itemgetter(1))[::-1]

        return vocab_coverage, text_coverage, sorted_missing

    @staticmethod
    def get_embedding_matrix(word_index, input_dim, glove_embeddings, glove_size):
        embedding_matrix = np.zeros((input_dim, glove_size))

        for word, index in word_index.items():

            embedding_vector = glove_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix

    @staticmethod
    def correct_data(data):
        data['target_relabeled'] = data['target'].copy()
        corrections = [
            (
                'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit',
                0),
            ('Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 0),
            ('To fight bioterrorism sir.', 0),
            (
                '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4',
                1),
            ('CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 1),
            (
                '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption',
                0),
            ('In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!',
             0),
            (
                'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE',
                1),
            ('RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 1),
            ("Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 0),
            ("wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 0),
            (
                "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam",
                0),
            (
                "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!",
                0),
            (
                "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'",
                0),
            ("Caution: breathing may be hazardous to your health.", 1),
            ("I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 0),
            (
                "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect",
                0),
            (
                "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time",
                0),
        ]

        for text, target in corrections:
            data.loc[data['text'] == text, 'target_relabeled'] = target

        ids_with_target_error = [328, 443, 513, 2619, 3640, 3900, 4342, 5781, 6552, 6554, 6570, 6701, 6702, 6729, 6861,
                                 7226]

        for t_id in ids_with_target_error:
            data.loc[data['id'] == t_id, 'target_relabeled'] = 0

    @staticmethod
    def get_bert_input(tweets, tokenizer, input_length=None):
        all_tokens = []
        all_masks = []
        all_segments = []

        tokenized_tweets = [tokenizer.tokenize(t) for t in tweets]

        max_len = Helpers.get_max_vector_len(tokenized_tweets) + 2 if input_length is None else input_length

        for tweet in tokenized_tweets:
            input_sequence = ["[CLS]"] + tweet[:max_len - 2] + ["[SEP]"]
            pad_len = max_len - len(input_sequence)
            tokens = tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        if input_length is None:
            return (np.array(all_tokens), np.array(all_masks), np.array(all_segments)), max_len

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    @staticmethod
    def get_model_inputs(data, tokenizer, input_len=None):
        tokenizer.fit_on_texts(data[0])

        sequences = []

        for data_set in data:
            sequences.append(tokenizer.texts_to_sequences(data_set))

        if input_len is None:
            input_len = Helpers.get_max_vector_len(sequences[0])

        inputs = []

        for sequence in sequences:
            inputs.append(pad_sequences(sequence, maxlen=input_len, truncating='post', padding='post'))

        input_dim = len(tokenizer.word_index) + 1

        return inputs, input_dim, input_len

    @staticmethod
    def with_glove_embedding_options(config, tokenizer, glove_embeddings):
        train_glove_vocab_coverage, train_glove_text_coverage, _ = Helpers.check_embeddings_coverage(
            tokenizer.word_counts,
            glove_embeddings
        )
        print(f'GloVe Vocab Coverage: {train_glove_vocab_coverage}; GloVe text Coverage: {train_glove_text_coverage}')
        config['GLOVE']['VOCAB_COVERAGE'] = train_glove_vocab_coverage
        config['GLOVE']['TEXT_COVERAGE'] = train_glove_text_coverage
        embedding_matrix = Helpers.get_embedding_matrix(
            word_index=tokenizer.word_index,
            input_dim=len(tokenizer.word_index) + 1,
            glove_embeddings=glove_embeddings,
            glove_size=config['GLOVE']['SIZE'],
        )
        config['EMBEDDING_OPTIONS']['output_dim'] = config['GLOVE']['SIZE']
        config['EMBEDDING_OPTIONS']['embeddings_initializer'] = Constant(embedding_matrix)
        config['EMBEDDING_OPTIONS']['trainable'] = False

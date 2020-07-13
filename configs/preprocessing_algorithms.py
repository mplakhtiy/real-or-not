PREPROCESSING_ALGORITHMS = {
    '1258a9d2-111e-4d4a-acda-852dd7ba3e88': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    '60314ef9-271d-4865-a7db-6889b1670f59': {
        'add_link_flag': False, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    '4c2e484d-5cb8-4e3e-ba7b-679ae7a73fca': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': True, 'add_location_flag': True, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    '8b7db91c-c8bf-40f2-986a-83a659b63ba6': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    '7bc816a1-25df-4649-8570-0012d0acd72a': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': False},
    'a85c8435-6f23-4015-9e8c-19547222d6ce': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False},
    'b054e509-4f04-44f2-bcf9-14fa8af4eeed': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False},
    '2e359f0b-bfb9-4eda-b2a4-cd839c122de6': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False},
    '71bd09db-e104-462d-887a-74389438bb49': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False},
    'd3cc3c6e-10de-4b27-8712-8017da428e41': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': False}
}


def get_preprocessing_algorithm(alg_id=None, join=False):
    if alg_id is None:
        return PREPROCESSING_ALGORITHMS.copy()

    for key, alg in PREPROCESSING_ALGORITHMS.items():
        if alg_id in key:
            if not join:
                return alg.copy()
            else:
                a = alg.copy()
                a['join'] = True

                return a

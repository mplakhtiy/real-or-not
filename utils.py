import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import time


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return type(obj).__name__


def plot(history, string, test_performance=None):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])

    if test_performance is not None and string == 'accuracy':
        plt.plot([float(v[:-1]) / 100 for v in test_performance])

    plt.xlabel("Epochs")
    plt.ylabel(string)
    if test_performance is not None and string == 'accuracy':
        plt.legend([string, 'val_' + string, 'test_accuracy'])
    else:
        plt.legend([string, 'val_' + string])
    plt.show()


def draw_keras_graph(history, test_performance=None):
    plot(history, "accuracy", test_performance)
    plot(history, 'loss')


def save_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf8') as file:
        file.write(json.dumps(data, ensure_ascii=False, default=dumper))


def get_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_prepared_data_from_file(file_path):
    data = get_from_file(file_path)

    return data.get('x_train'), data.get('y_train'), data.get('x_val'), data.get('y_val'), data


def log_execution_time(start_time):
    print("--- %.2f%%s seconds ---" % (time.time() - start_time))


def log(
        file=None,
        data=None,
        vocabulary=None,
        model=None,
        model_history=None,
        model_config=None,
        test_performance=None,
        classifier=None,
        vectorizer=None,
        file_path=f'./logs/log-{datetime.now().date()}.json'
):
    if not os.path.exists(file_path):
        save_to_file(file_path, {})

    log_data = get_from_file(file_path)

    current_log = {}
    if file is not None:
        current_log['file'] = file

    if data is not None:
        current_log['data'] = data

    if vocabulary is not None:
        current_log['vocabulary']: vocabulary

    if model is not None:
        current_log['model'] = model
        current_log['model']['config'] = model_config
        current_log['model']['val_accuracy'] = [float(a) for a in model_history['val_accuracy']]
        current_log['model']['test_performance'] = test_performance

    if classifier is not None:
        current_log['classifier'] = classifier

    if vectorizer is not None:
        current_log['vectorizer'] = vectorizer

    log_data[str(datetime.now())] = current_log

    save_to_file(file_path, log_data)


def get_glove_from_txt(file_path):
    embeddings_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector.tolist()

    return embeddings_dict


def relable(data):
    data['target_relabeled'] = data['target'].copy()

    data.loc[data[
                 'text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_relabeled'] = 0
    data.loc[data['text'] == 'To fight bioterrorism sir.', 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_relabeled'] = 1
    data.loc[data[
                 'text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_relabeled'] = 1
    data.loc[data[
                 'text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_relabeled'] = 1
    data.loc[data[
                 'text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_relabeled'] = 1
    data.loc[data[
                 'text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_relabeled'] = 0
    data.loc[data['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_relabeled'] = 0
    data.loc[data['text'] == "Caution: breathing may be hazardous to your health.", 'target_relabeled'] = 1
    data.loc[data[
                 'text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_relabeled'] = 0
    data.loc[data[
                 'text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_relabeled'] = 0

    return data

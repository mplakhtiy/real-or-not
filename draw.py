from models.keras import Keras
from utils import get_from_file

history = get_from_file('./round_table_history.json')

print(max(history['test_accuracy']))

Keras.draw_graph(history)

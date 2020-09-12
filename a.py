# fa = [
#     103,
#     20,
#     9,
#     30,
#     42,
#     37,
#     29,
#     39,
#     20,
#     14,
#     29,
#     63,
#     43,
#     51,
#     46,
#     4,
#     51,
#     41,
#     41,
#     43,
#     43,
#     12,
#     14,
#     7,
#     19,
#     90,
#     22,
# ]
# gue = [
#     9192,
#     8952,
#     9021,
#     8952,
#     8906,
#     8978,
#     8912,
#     8940,
#     8942,
#     9005,
#     9011,
#     8874,
#     8856,
#     8840,
#     8858,
#     8963,
#     8829,
#     8922,
#     8830,
#     8854,
#     8840,
#     8909,
#     8914,
#     8926,
#     8879,
#     8589,
#     8749,
# ]
# res = []
# for i in range(27):
#     res.append(round(((fa[i] / 10) + gue[i]) / 10876, 5))

# tds_1 = [
#     0.84054,
#     0.81492,
#     0.81978,
#     0.81610,
#     0.81059,
#     0.81821,
#     0.81532,
#     0.81138,
#     0.81571,
#     0.82320,
#     0.82070,
#     0.80349,
#     0.80454,
#     0.80494,
#     0.79929,
#     0.81610,
#     0.80284,
#     0.80533,
#     0.80888,
#     0.80139,
#     0.80599,
#     0.80796,
#     0.81689,
#     0.81229,
#     0.80704,
#     0.77144,
#     0.79771
# ]
#
# tds_2 = [
#     0.83745,
#     0.81937,
#     0.82195,
#     0.81889,
#     0.81375,
#     0.82326,
#     0.81846,
#     0.82036,
#     0.81693,
#     0.82395,
#     0.82483,
#     0.81250,
#     0.81342,
#     0.80642,
#     0.81077,
#     0.81957,
#     0.81019,
#     0.81730,
#     0.81520,
#     0.80981,
#     0.81538,
#     0.81618,
#     0.81635,
#     0.81797,
#     0.81340,
#     0.78634,
#     0.80411
# ]
#
# res = []
# for i in range(27):
#     res.append(round(tds_2[i] - tds_1[i], 5))
#
# print(res)
#

from utils import get_from_file

log = get_from_file('./logs/bert/failed/BERT-347-failed.json')


def get_best_epoch(history):
    best_loss_index = history['val_loss'].index(min(history['val_loss']))
    temp_acc = 0
    temp_loss = 999999999
    best_acc_index = 0

    for i in range(best_loss_index + 1):
        if temp_acc <= history['val_accuracy'][i] and history['val_loss'][i] <= temp_loss:
            temp_acc = history['val_accuracy'][i]
            temp_loss = history['val_loss'][i]
            best_acc_index = i

    return best_acc_index, best_loss_index


l = [0, 0, 0]

for fold in list(log.values())[0]['KFOLD_HISTORY']:
    best_acc_index, best_loss_index = get_best_epoch(fold)
    l[0] = l[0] + fold['val_accuracy'][best_acc_index]
    l[1] = l[1] + fold['train_set_accuracy'][best_acc_index]
    l[2] = l[2] + fold['test_set_accuracy'][best_acc_index]
print(l[0]/10, l[1]/10, l[2]/10)

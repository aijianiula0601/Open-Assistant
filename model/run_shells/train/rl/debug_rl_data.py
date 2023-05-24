import json

org_f = "/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/debug_multi_dataset_qas.json"
save_f = "/mnt/cephfs/hjh/train_record/nlp/open-assistant/dataset/debug_my_rl.json"

data = json.load(open(org_f))


debug_data = []
for item in data:

    cur_qas = []

    for i in range(len(item['qas'])):
        qa = item['qas'][f'turn_{i}']
        cur_qas.append(qa['question'])
        cur_qas.append(qa['answer'])

    debug_data.append(cur_qas)

json.dump(debug_data, open(save_f, 'w'))
print(f"save to:{save_f}")

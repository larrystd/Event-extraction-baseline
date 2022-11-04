#! /usr/bin/env python
import json
import os
import random
import statistics
import logging
from collections import Counter, defaultdict

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGPATH = "./chfin/data_statistics.log"
if os.path.exists(LOGPATH):
    os.remove(LOGPATH)

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG,
    filename=LOGPATH,
    filemode='a')

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
mpl.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams["axes.titlesize"] = 20

SEED = 403
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

DATASET_DIR = "./Data"
PROCESSED_DATA_DIR = "./chfin"

# hbar
def barh(label2num: dict, title: str = "No Title", limit=None):
    label2num_sorted = sorted(label2num.items(), key=lambda x: x[1])
    if limit:
        label2num_sorted = label2num_sorted[:limit]
    tot = sum([x[1] for x in label2num_sorted])
    fig = plt.figure(figsize=(16, 9), dpi=350)
    ax = fig.add_subplot(111)
    ax.barh(range(len(label2num_sorted)), [x[1] for x in label2num_sorted], zorder=3)
    ax.set_yticks(range(len(label2num_sorted)))
    ax.set_yticklabels(
        [
            "{} - {} ({:.2f}%)".format(x[0], x[1], float(x[1]) / tot * 100)
            for x in label2num_sorted
        ],
        fontsize=16,
    )
    ax.set_xlabel("Total: {}".format(tot), fontsize=16)
    ax.set_title(title)
    ax.grid(zorder=0)
    plt.rc("axes", axisbelow=True)
    plt.rc("ytick", labelsize=16)
    plt.tight_layout()
    # plt.show()
    fig.savefig(f"{title}.png", format="png")
    plt.close()


# hist
def hist(data: list, bins: int = 100, title: str = "No Title", threshold=0.9):
    tot = sum(data)
    fig = plt.figure(figsize=(16, 9), dpi=350)
    ax = fig.add_subplot(111)
    recs = ax.hist(data, bins=bins, rwidth=0.8, zorder=3)
    tot_num = recs[0].sum()
    nums = 0
    i = 0
    for num in recs[0]:
        nums += num
        if nums / tot_num >= threshold:
            break
        i += 1
    ax.vlines(recs[1][i], 0, recs[0].max(), colors="r", linestyles="dashed")
    min_data = min(data)
    mean_data = sum(data) / len(data)
    max_data = max(data)
    ax.set_xlabel(
        "Total: {}, Min: {}, Mean: {:.3f}, Max: {} - Vline: {:.3f} ({:.2f}%)".format(
            tot, min_data, mean_data, max_data, recs[1][i], nums / tot_num * 100
        ),
        fontsize=16,
    )
    ax.set_title(title)
    ax.grid(zorder=0)
    plt.rc("axes", axisbelow=True)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"{title}.png", format="png")
    plt.close()


def load_json(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin)
    return data


def dump_json(obj, filepath, **kwargs):
    with open(filepath, "wt", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, **kwargs)


def stat(data, dataset_name, plot_figures=True):
    # data从json文件中读取到的json对象
    logging.info(f"============== stat on {dataset_name} ==============")
    num_doc = len(data)  # number of document
    logging.info(f"#docs: {num_doc}")
    (
        in_events_span,
        len_sent,
        len_docs,
        num_sent,
        num_span,
        num_evt,
        evt_types,
        num_o2o_doc,
        o2o_evt_types,
        num_o2m_doc,
        o2m_evt_types,
        num_m2m_doc,
        m2m_evt_types,
        one_doc_eventtypes_num,
        one_doc_events_num,
        doc_has_max_events,
        doc_has_max_events_id,
    ) = ([], [], [], [], [], [], {}, 0, {}, 0, {}, 0, {}, {}, {}, {},{})

    event_roles = defaultdict(set)
    args_stat = defaultdict(list)
    for d in data:
        # d is json object of one doc where d[0]: id, d[1]: content
        doc = d[1]
        doc["doc_type"] = ""  # o2o, o2m, m2m
        num_span.append(len(doc["ann_valid_mspans"]))  # number of candidate argument 
        num_sent.append(len(doc["sentences"]))  # number of sentence
        len_sent_tmp = list(map(len, doc["sentences"]))  # len of each sentence
        len_docs.append(sum(len_sent_tmp))  # len of document
        len_sent.extend(len_sent_tmp)  # append list
        events = doc["recguid_eventname_eventdict_list"]  # [0, 'EquityPledge', {'Pledger': '李华青', 'PledgedShares':
        num_evt.append(len(events))
        evttype_list = [e[1] for e in events]  # event type list ['EquityPledge', 'EquityPledge']
        evttype_counter = Counter(evttype_list)  # number of every event type, like {'EquityPledge': 2}
        if len(evttype_counter) == 1:  # only one event type
            # document classify three types: o2o, o2m, m2m
            event, num = evttype_counter.most_common()[0]
            if 1 not in one_doc_eventtypes_num:
                one_doc_eventtypes_num[1] = 1
            else:
                one_doc_eventtypes_num[1] += 1 
            if num == 1:  # one event,
                if event not in evt_types:  # o2o(one-one) type document
                    evt_types[event] = 1 
                else:
                    evt_types[event] += 1

                if event not in one_doc_events_num:
                    one_doc_events_num[event] = {}
                    one_doc_events_num[event][1] = 1
                elif 1 not in one_doc_events_num[event]:
                    one_doc_events_num[event][1] = 1
                else:
                   one_doc_events_num[event][1] += 1 
                    
                if event not in o2o_evt_types:
                    o2o_evt_types[event] = 1
                else:
                    o2o_evt_types[event] += 1
                if event not in doc_has_max_events:
                    doc_has_max_events[event] = 1
                    doc_has_max_events_id[event] = d[0]
                num_o2o_doc += 1
                doc["doc_type"] = "o2o"
            elif num > 1:  # one event type, multi events, o2m(one-multi) type document
                if event not in evt_types:
                    evt_types[event] = num
                else:
                    evt_types[event] += num
                if event not in one_doc_events_num:
                    one_doc_events_num[event] = {}
                    one_doc_events_num[event][num] = 1
                elif num not in one_doc_events_num[event]:
                    one_doc_events_num[event][num] = 1
                else:
                   one_doc_events_num[event][num] += 1

                if event not in o2m_evt_types:
                    o2m_evt_types[event] = num
                else:
                    o2m_evt_types[event] += num
                if event not in doc_has_max_events:
                    doc_has_max_events[event] = num
                    doc_has_max_events_id[event] = d[0]
                elif doc_has_max_events[event] < num:
                    doc_has_max_events[event] = num
                    doc_has_max_events_id[event] = d[0]
                num_o2m_doc += 1
                doc["doc_type"] = "o2m"
            else:
                breakpoint()  # trap in pdb
        elif len(evttype_counter) > 1:
            num_m2m_doc += 1
            if len(evttype_counter) not in one_doc_eventtypes_num:
                one_doc_eventtypes_num[len(evttype_counter)] = 1
            else:
                one_doc_eventtypes_num[len(evttype_counter)] += 1 

            for event, num in evttype_counter.items():  # # m2m(multi-multi) type document
                if event not in evt_types:
                    evt_types[event] = num
                else:
                    evt_types[event] += num

                if event not in one_doc_events_num:
                    one_doc_events_num[event] = {}
                    one_doc_events_num[event][num] = 1
                elif num not in one_doc_events_num[event]:
                    one_doc_events_num[event][num] = 1
                else:
                   one_doc_events_num[event][num] += 1
                
                if event not in m2m_evt_types:
                    m2m_evt_types[event] = num
                else:
                    m2m_evt_types[event] += num
                if event not in doc_has_max_events:
                    doc_has_max_events[event] = num
                elif doc_has_max_events[event] < num:
                    doc_has_max_events[event] = num
                    doc_has_max_events_id[event] = d[0]
                doc["doc_type"] = "m2m"
        else:
            breakpoint()
        # process event type of document
        doc_in_events_span = set()
        for event in events:
            event_type = event[1]
            tmp_roles = set()
            for role in event[2]:  # event[2] role map
                event_roles[event_type].add(role)
                if event[2][role] is not None:  # argument is not None, add roles to tmp_roles
                    tmp_roles.add(role)
            args_stat[event_type].append(tmp_roles)
            for _, span in event[2].items():
                doc_in_events_span.add(span)  # span or role
        in_events_span.append(len(doc_in_events_span))

    logging.info(f"avg #sent per doc: {sum(num_sent) / num_doc}")
    if plot_figures:
        hist(num_sent, title=f"{dataset_name}_num_sent_per_doc")
        hist(len_sent, title=f"{dataset_name}_len_sent")
        hist(len_docs, title=f"{dataset_name}_len_doc")
    logging.info(
        "The number of spans: AVG: {}, MAX: {}, MEDIAN: {}".format(
            sum(num_span) / len(num_span), max(num_span), statistics.median(num_span)
        )
    )
    if plot_figures:
        hist(num_span, title=f"{dataset_name}_num_span_per_doc")
    logging.info(
        "#in_events_span: avg: {}, max: {}, median: {}".format(
            sum(in_events_span) / len(in_events_span),
            max(in_events_span),
            statistics.median(in_events_span),
        )
    )
    if plot_figures:
        hist(in_events_span, title=f"{dataset_name}_in_events_span")
    num_evt_counter = Counter(num_evt)
    logging.info(f"#events: {sum(num_evt)}, MAX number of events: {max(num_evt)}")
    if plot_figures:
        barh(num_evt_counter, title=f"{dataset_name}_num_event_per_doc")
    logging.warning(f"Number of each event type")
    for event_type in evt_types:
        logging.info( 
            f"event_type: {event_type}, number, {evt_types[event_type]}"
        )
    logging.info(f"len of o2o doc: {num_o2o_doc}")
    logging.warning(f"Number of each event type in o2o doc")
    for event_type in o2o_evt_types:
        logging.info( 
            f"event_type: {event_type}, number, {o2o_evt_types[event_type]}"
        )
    logging.info(f"len of o2m doc: {num_o2m_doc}")
    logging.warning(f"Number of each event type in o2m doc")
    for event_type in o2m_evt_types:
        logging.info( 
            f"event_type: {event_type}, number, {o2m_evt_types[event_type]}"
        )
    logging.info(f"len of m2m doc: {num_m2m_doc}")
    logging.warning(f"Number of each event type in m2m doc")
    for event_type in m2m_evt_types:
        logging.info( 
            f"event_type: {event_type}, number, {m2m_evt_types[event_type]}"
        )
    
    logging.warning(f"Number of eventtypes in one doc")
    for num in one_doc_eventtypes_num:
        logging.info( 
            f"{num}: {one_doc_eventtypes_num[num]}"
        )

    logging.warning(f"Number of each event type in one doc")
    for event_type in one_doc_events_num:
        logging.info( 
            f"event_type: {event_type}"
        )  
        for num in one_doc_events_num[event_type]:
            logging.info( 
                f"{num}: {one_doc_events_num[event_type][num]}"
            )

    logging.info("event_type2args:")
    for event_type in event_roles:  # map: event_type->roles list
        logging.info(
            f"{event_type},{len(args_stat[event_type])},{len(event_roles[event_type])},{event_roles[event_type]}"
        )
        args_num = [len(x) for x in args_stat[event_type]]
        logging.info(
            f"{event_type}: min_arg_num: {min(args_num)}, avg_arg_num: {sum(args_num) / len(args_num)}, median: {statistics.median(args_num)}"
        )
        arg_num_counter = Counter(args_num)
        if plot_figures:
            barh(arg_num_counter, title=f"{dataset_name}_{event_type}_arg_num")
        num_events = len(args_num)
        all_args = []
        for args in args_stat[event_type]:
            all_args.extend(args)
        all_args_counter = Counter(all_args)
        for arg_name in all_args_counter:
            all_args_counter[arg_name] /= num_events
        if plot_figures:
            barh(all_args_counter, title=f"{dataset_name}_{event_type}_arg_type_count")
    dump_json(data, f"{PROCESSED_DATA_DIR}/typed_{dataset_name}.json")


def stat_ent_part(filename):
    """how many ents participate in the final instances?"""
    logging.info("====================================================")
    logging.info(filename)
    data = load_json(filename)
    num_ents = {"o2o": 0, "o2m": 0, "m2m": 0, "overall": 0}
    num_part_ents = {"o2o": 0, "o2m": 0, "m2m": 0, "overall": 0}
    for d in data:
        d = d[1]
        doc_type = d["doc_type"]
        ents = set(d["ann_valid_mspans"])
        part_ents = set()
        for ins in d["recguid_eventname_eventdict_list"]:
            part_ents.update(set(filter(lambda x: x is not None, ins[2].values())))
        if len(part_ents - ents) > 0:
            breakpoint()
        num_ents[doc_type] += len(ents)
        num_part_ents[doc_type] += len(part_ents)
    num_ents["overall"] = num_ents["o2o"] + num_ents["o2m"] + num_ents["m2m"]
    num_part_ents["overall"] = (
        num_part_ents["o2o"] + num_part_ents["o2m"] + num_part_ents["m2m"]
    )
    for data_type in num_ents:
        logging.info(
            f"{data_type}: {num_part_ents[data_type]}/{num_ents[data_type]} = "
            + "{:.3f}".format(num_part_ents[data_type] / num_ents[data_type] * 100)
        )


def stat_doc_type(data):
    type2cnt = defaultdict(lambda: {"o2o": 0, "o2m": 0, "m2m": 0})
    for _, doc in data:
        doc_type = doc["doc_type"]
        for _, ins_type, _ in doc["recguid_eventname_eventdict_list"]:
            type2cnt["Overall"][doc_type] += 1
            type2cnt[ins_type][doc_type] += 1
    for doc_type in type2cnt:
        print(doc_type, type2cnt[doc_type])


def stat_len(data):
    char_lens = []
    num_sents = []
    sent_lens = []
    for d in data:
        d = d[1]
        sents = d["sentences"]
        num_sents.append(len(sents))
        sent_len = [len(s) for s in sents]
        sent_lens.extend(sent_len)
        char_lens.append(sum(sent_len))
    logging.info(
        f"num_sents: avg: {statistics.mean(num_sents)}, median: {statistics.median(num_sents)}, max: {max(num_sents)}"
    )
    logging.info(
        f"sent_lens: avg: {statistics.mean(sent_lens)}, median: {statistics.median(sent_lens)}, max: {max(sent_lens)}"
    )
    logging.info(
        f"char_lens: avg: {statistics.mean(char_lens)}, median: {statistics.median(char_lens)}, max: {max(char_lens)}"
    )


def load_line_json(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            data.append(json.loads(line))
    return data


def stat_len_duee_fin(data):
    char_lens = []
    num_sents = []
    sent_lens = []
    for d in data:
        sents = d["text"].split("\n")
        num_sents.append(len(sents))
        sent_len = [len(s) for s in sents]
        sent_lens.extend(sent_len)
        char_lens.append(sum(sent_len))
    logging.info(
        f"num_sents: avg: {statistics.mean(num_sents)}, median: {statistics.median(num_sents)}, max: {max(num_sents)}"
    )
    logging.info(
        f"sent_lens: avg: {statistics.mean(sent_lens)}, median: {statistics.median(sent_lens)}, max: {max(sent_lens)}"
    )
    logging.info(
        f"char_lens: avg: {statistics.mean(char_lens)}, median: {statistics.median(char_lens)}, max: {max(char_lens)}"
    )


def stat_argument_scattering_docs(data):
    num_ins = 0
    same_sent_arg_ins = 0
    num_args = 0
    num_scattered_args = 0
    for d in data:
        doc = d[1]
        for _, _, role2args in doc["recguid_eventname_eventdict_list"]:
            num_ins += 1
            arg_dranges = []
            for role, arg in role2args.items():
                if arg is not None:
                    dranges = doc["ann_mspan2dranges"][arg]
                    arg_dranges.extend(dranges)
                    num_args += 1
                    if len(dranges) > 1:
                        num_scattered_args += 1
            if len(set([x[0] for x in arg_dranges])) == 1:
                same_sent_arg_ins += 1
    logging.info(f"num_ins: {num_ins}, same_sent_arg_ins: {same_sent_arg_ins}")
    logging.info(f"num_args: {num_args}, num_scattered_args: {num_scattered_args}")


if __name__ == "__main__":
    if not os.path.exists(f"{PROCESSED_DATA_DIR}"):
        os.mkdir(f"{PROCESSED_DATA_DIR}")
    for data_name in ["train", "dev", "test"]:
        logging.warning(
            f"{data_name}!"
        )
        data = load_json(f"{DATASET_DIR}/{data_name}.json")  # f结合{} 格式化成字符串
        stat(data, data_name, plot_figures=False)

    # select 48 from sample_train for debug
    data = load_json(f"{DATASET_DIR}/sample_train.json")
    data = [random.choice(data) for _ in range(48)]
    stat(data, "sample_train_48", plot_figures=False)

    # # 1/8
    # data = load_json("train.json")
    # random.shuffle(data)
    # data = data[:len(data) // 8]
    # stat(data, "train_1o8", plot_figures=False)

    # # select 1/4 documents from train
    # data = load_json("train.json")
    # random.shuffle(data)
    # data = data[:len(data) // 4]
    # stat(data, "train_1o4", plot_figures=False)  # 1 over 4

    # # half
    # data = load_json("train.json")
    # random.shuffle(data)
    # data = data[:len(data) // 2]
    # stat(data, "train_1o2", plot_figures=False)  # 1 over 2

    # data = load_json("typed_train_1o4.json")
    # stat(data, "typed_train_1o4")

    # one = load_json("typed_test.json")
    # random.shuffle(one)
    # sample = None
    # for d in one:
    #     if d[1]['doc_type'] == 'o2m':
    #         sample = d
    #         break
    # dump_json(sample, "o2m_sample_one.json", indent=2)

    # for filepath in ["typed_train.json", "typed_dev.json", "typed_test.json"]:
    #     stat_ent_part(filepath)

    # data = load_json("typed_train.json")
    # stat_doc_type(data)
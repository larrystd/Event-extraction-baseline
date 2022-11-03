import copy
import re
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from dee.utils import default_load_json, logger, regex_extractor


class DEEExample(object):
    def __init__(
        self,
        annguid,
        detail_align_dict,
        template,
        tokenizer,
        only_inference=False,
        inlcude_complementary_ents=False,
    ):
        self.guid = annguid
        # [sent_text, ...]
        self.sentences = detail_align_dict["sentences"]
        self.doc_type = detail_align_dict.get("doc_type", "unk")
        self.num_sentences = len(self.sentences)
        if inlcude_complementary_ents:
            self.complementary_field2ents = regex_extractor.extract_doc(
                detail_align_dict["sentences"]
            )
        else:
            self.complementary_field2ents = {}

        if only_inference:
            # set empty entity/event information
            self.only_inference = True
            self.ann_valid_mspans = []
            self.ann_mspan2dranges = {}
            self.ann_mspan2guess_field = {}
            self.recguid_eventname_eventdict_list = []
            self.num_events = 0
            self.sent_idx2srange_mspan_mtype_tuples = {}
            self.event_type2event_objs = {}
        else:
            # set event information accordingly
            self.only_inference = False

            if inlcude_complementary_ents:
                # build index
                comp_ents_sent_index = defaultdict(list)
                comp_ents_start_index = defaultdict(list)
                comp_ents_end_index = defaultdict(list)
                for raw_field, ents in self.complementary_field2ents.items():
                    # field = 'Other' + field.title()
                    field = "OtherType"
                    for ent, pos_span in ents:
                        pos_span = list(pos_span)
                        if ent not in detail_align_dict["ann_valid_mspans"]:
                            comp_ents_sent_index[pos_span[0]].append(
                                [ent, raw_field, pos_span]
                            )
                            comp_ents_start_index[(pos_span[0], pos_span[1])].append(
                                [ent, raw_field, pos_span]
                            )
                            comp_ents_end_index[(pos_span[0], pos_span[2])].append(
                                [ent, raw_field, pos_span]
                            )

                # remove overlaped Date
                mspan2fields = copy.deepcopy(detail_align_dict["ann_mspan2guess_field"])
                mspan2dranges = copy.deepcopy(detail_align_dict["ann_mspan2dranges"])
                for ent, field in mspan2fields.items():
                    for drange in mspan2dranges[ent]:
                        for s_ent in comp_ents_sent_index.get(drange[0], []):
                            s_ent, raw_field, pos_span = s_ent
                            if (
                                drange[1] <= pos_span[1] < drange[2]
                                or drange[1] < pos_span[2] <= drange[2]
                            ):
                                if [s_ent, pos_span] in self.complementary_field2ents[
                                    raw_field
                                ]:
                                    self.complementary_field2ents[raw_field].remove(
                                        [s_ent, pos_span]
                                    )

                for raw_field, ents in self.complementary_field2ents.items():
                    field = "OtherType"
                    for ent, pos_span in ents:
                        pos_span = list(pos_span)
                        if ent not in detail_align_dict["ann_valid_mspans"]:
                            detail_align_dict["ann_valid_mspans"].append(ent)
                            detail_align_dict["ann_mspan2guess_field"][ent] = field
                            detail_align_dict["ann_mspan2dranges"][ent] = [pos_span]
                        elif (
                            list(pos_span)
                            not in detail_align_dict["ann_mspan2dranges"][ent]
                        ):
                            detail_align_dict["ann_mspan2dranges"][ent].append(pos_span)

                # OtherType wrong ratio annotation correction
                mspan2fields = copy.deepcopy(detail_align_dict["ann_mspan2guess_field"])
                mspan2dranges = copy.deepcopy(detail_align_dict["ann_mspan2dranges"])
                for ent, field in mspan2fields.items():
                    if field == "OtherType" and "%" in ent:
                        for drange in mspan2dranges[ent]:
                            if self.sentences[drange[0]][drange[1] - 1] in "0123456789":
                                # not-complete ratio, drop
                                detail_align_dict["ann_valid_mspans"].remove(ent)
                                detail_align_dict["ann_mspan2guess_field"].pop(ent)
                                detail_align_dict["ann_mspan2dranges"].pop(ent)
                                break

            # [span_text, ...]
            self.ann_valid_mspans = detail_align_dict["ann_valid_mspans"]
            # span_text -> [drange_tuple, ...]
            self.ann_mspan2dranges = detail_align_dict["ann_mspan2dranges"]
            # span_text -> guessed_field_name
            self.ann_mspan2guess_field = detail_align_dict["ann_mspan2guess_field"]
            # [(recguid, event_name, event_dict), ...]
            self.recguid_eventname_eventdict_list = detail_align_dict[
                "recguid_eventname_eventdict_list"
            ]
            self.num_events = len(self.recguid_eventname_eventdict_list)

            # for create ner examples
            # sentence_index -> [(sent_match_range, match_span, match_type), ...]
            self.sent_idx2srange_mspan_mtype_tuples = {}
            for sent_idx in range(self.num_sentences):
                self.sent_idx2srange_mspan_mtype_tuples[sent_idx] = []

            for mspan in self.ann_valid_mspans:
                for drange in self.ann_mspan2dranges[mspan]:
                    sent_idx, char_s, char_e = drange
                    sent_mrange = (char_s, char_e)

                    sent_text = self.sentences[sent_idx]
                    sent_text = tokenizer.dee_tokenize(sent_text)
                    if sent_text[char_s:char_e] != tokenizer.dee_tokenize(mspan):
                        raise ValueError(
                            "GUID: {} span range is not correct, span={}, range={}, sent={}".format(
                                annguid, mspan, str(sent_mrange), sent_text
                            )
                        )

                    guess_field = self.ann_mspan2guess_field[mspan]

                    self.sent_idx2srange_mspan_mtype_tuples[sent_idx].append(
                        (sent_mrange, mspan, guess_field)
                    )

            # for create event objects the length of event_objs should >= 1
            self.event_type2event_objs = {}
            for (
                mrecguid,
                event_name,
                event_dict,
            ) in self.recguid_eventname_eventdict_list:
                event_class = template.event_type2event_class[event_name]
                event_obj = event_class()
                # assert isinstance(event_obj, BaseEvent)
                event_obj.update_by_dict(event_dict, recguid=mrecguid)

                if event_obj.name in self.event_type2event_objs:
                    self.event_type2event_objs[event_obj.name].append(event_obj)
                else:
                    self.event_type2event_objs[event_name] = [event_obj]

    def __repr__(self):
        dee_str = "DEEExample (\n"
        dee_str += "  guid: {},\n".format(repr(self.guid))

        if not self.only_inference:
            dee_str += "  span info: (\n"
            for span_idx, span in enumerate(self.ann_valid_mspans):
                gfield = self.ann_mspan2guess_field[span]
                dranges = self.ann_mspan2dranges[span]
                dee_str += "    {:2} {:20} {:30} {}\n".format(
                    span_idx, span, gfield, str(dranges)
                )
            dee_str += "  ),\n"

            dee_str += "  event info: (\n"
            event_str_list = repr(self.event_type2event_objs).split("\n")
            for event_str in event_str_list:
                dee_str += "    {}\n".format(event_str)
            dee_str += "  ),\n"

        dee_str += "  sentences: (\n"
        for sent_idx, sent in enumerate(self.sentences):
            dee_str += "    {:2} {}\n".format(sent_idx, sent)
        dee_str += "  ),\n"

        dee_str += ")\n"

        return dee_str


class DEEExampleLoader(object):
    def __init__(
        self,
        template,
        tokenizer,
        rearrange_sent_flag,
        max_sent_len,
        drop_irr_ents_flag=False,
        include_complementary_ents=False,
        filtered_data_types=["o2o", "o2m", "m2m"],
    ):
        self.template = template
        self.tokenizer = tokenizer
        self.rearrange_sent_flag = rearrange_sent_flag
        self.max_sent_len = max_sent_len
        self.drop_irr_ents_flag = drop_irr_ents_flag
        self.include_complementary_ents_flag = include_complementary_ents
        self.filtered_data_types = filtered_data_types

    def rearrange_sent_info(self, detail_align_info):
        if "ann_valid_dranges" not in detail_align_info:
            detail_align_info["ann_valid_dranges"] = []
        if "ann_mspan2dranges" not in detail_align_info:
            detail_align_info["ann_mspan2dranges"] = {}

        detail_align_info = dict(detail_align_info)
        split_rgx = re.compile("[，：:；;）)]")

        raw_sents = detail_align_info["sentences"]
        doc_text = "".join(raw_sents)
        raw_dranges = detail_align_info["ann_valid_dranges"]
        raw_sid2span_char_set = defaultdict(lambda: set())
        for raw_sid, char_s, char_e in raw_dranges:
            span_char_set = raw_sid2span_char_set[raw_sid]
            span_char_set.update(range(char_s, char_e))

        # try to split long sentences into short ones by comma, colon, semi-colon, bracket
        short_sents = []
        for raw_sid, sent in enumerate(raw_sents):
            span_char_set = raw_sid2span_char_set[raw_sid]
            if len(sent) > self.max_sent_len:
                cur_char_s = 0
                for mobj in split_rgx.finditer(sent):
                    m_char_s, m_char_e = mobj.span()
                    if m_char_s in span_char_set:
                        continue
                    short_sents.append(sent[cur_char_s:m_char_e])
                    cur_char_s = m_char_e
                short_sents.append(sent[cur_char_s:])
            else:
                short_sents.append(sent)

        # merge adjacent short sentences to compact ones that match max_sent_len
        comp_sents = [""]
        for sent in short_sents:
            prev_sent = comp_sents[-1]
            if len(prev_sent + sent) <= self.max_sent_len:
                comp_sents[-1] = prev_sent + sent
            else:
                comp_sents.append(sent)

        # get global sentence character base indexes
        raw_char_bases = [0]
        for sent in raw_sents:
            raw_char_bases.append(raw_char_bases[-1] + len(sent))
        comp_char_bases = [0]
        for sent in comp_sents:
            comp_char_bases.append(comp_char_bases[-1] + len(sent))

        assert raw_char_bases[-1] == comp_char_bases[-1] == len(doc_text)

        # calculate compact doc ranges
        raw_dranges.sort()
        raw_drange2comp_drange = {}
        prev_comp_sid = 0
        for raw_drange in raw_dranges:
            raw_drange = tuple(
                raw_drange
            )  # important when json dump change tuple to list
            raw_sid, raw_char_s, raw_char_e = raw_drange
            raw_char_base = raw_char_bases[raw_sid]
            doc_char_s = raw_char_base + raw_char_s
            doc_char_e = raw_char_base + raw_char_e
            assert doc_char_s >= comp_char_bases[prev_comp_sid]

            cur_comp_sid = prev_comp_sid
            for cur_comp_sid in range(prev_comp_sid, len(comp_sents)):
                if doc_char_e <= comp_char_bases[cur_comp_sid + 1]:
                    prev_comp_sid = cur_comp_sid
                    break
            comp_char_base = comp_char_bases[cur_comp_sid]
            assert (
                comp_char_base
                <= doc_char_s
                < doc_char_e
                <= comp_char_bases[cur_comp_sid + 1]
            )
            comp_char_s = doc_char_s - comp_char_base
            comp_char_e = doc_char_e - comp_char_base
            comp_drange = (cur_comp_sid, comp_char_s, comp_char_e)

            raw_drange2comp_drange[raw_drange] = comp_drange
            assert (
                raw_sents[raw_drange[0]][raw_drange[1] : raw_drange[2]]
                == comp_sents[comp_drange[0]][comp_drange[1] : comp_drange[2]]
            )

        # update detailed align info with rearranged sentences
        detail_align_info["sentences"] = comp_sents
        detail_align_info["ann_valid_dranges"] = [
            raw_drange2comp_drange[tuple(raw_drange)]
            for raw_drange in detail_align_info["ann_valid_dranges"]
        ]
        ann_mspan2comp_dranges = {}
        for ann_mspan, mspan_raw_dranges in detail_align_info[
            "ann_mspan2dranges"
        ].items():
            comp_dranges = [
                raw_drange2comp_drange[tuple(raw_drange)]
                for raw_drange in mspan_raw_dranges
            ]
            ann_mspan2comp_dranges[ann_mspan] = comp_dranges
        detail_align_info["ann_mspan2dranges"] = ann_mspan2comp_dranges

        return detail_align_info

    def drop_irr_ents(self, detail_align_info):
        ann_valid_mspans = []
        ann_valid_dranges = []
        ann_mspan2dranges = {}
        ann_mspan2guess_field = {}

        real_valid_spans = set()
        for _, _, role2span in detail_align_info["recguid_eventname_eventdict_list"]:
            spans = set(role2span.values())
            real_valid_spans.update(spans)
        if None in real_valid_spans:
            real_valid_spans.remove(None)
        for span in real_valid_spans:
            ann_valid_mspans.append(span)
            ann_valid_dranges.extend(detail_align_info["ann_mspan2dranges"][span])
            ann_mspan2dranges[span] = detail_align_info["ann_mspan2dranges"][span]
            ann_mspan2guess_field[span] = detail_align_info["ann_mspan2guess_field"][
                span
            ]

        detail_align_info["ann_valid_mspans"] = ann_valid_mspans
        detail_align_info["ann_valid_dranges"] = ann_valid_dranges
        detail_align_info["ann_mspan2dranges"] = ann_mspan2dranges
        detail_align_info["ann_mspan2guess_field"] = ann_mspan2guess_field
        return detail_align_info

    def convert_dict_to_example(self, annguid, detail_align_info, only_inference=False):
        if self.drop_irr_ents_flag:
            detail_align_info = self.drop_irr_ents(detail_align_info)
        if self.rearrange_sent_flag:
            detail_align_info = self.rearrange_sent_info(detail_align_info)
        dee_example = DEEExample(
            annguid,
            detail_align_info,
            self.template,
            self.tokenizer,
            only_inference=only_inference,
            inlcude_complementary_ents=self.include_complementary_ents_flag,
        )

        return dee_example

    def __call__(self, dataset_json_path, only_inference=False):
        """for every document generate a DEEExample object, return list(DEEExample)
        """
        total_dee_examples = []
        annguid_aligninfo_list = default_load_json(dataset_json_path)
        for annguid, detail_align_info in annguid_aligninfo_list:
            if detail_align_info["doc_type"] not in self.filtered_data_types:
                continue
            dee_example = self.convert_dict_to_example(
                annguid, detail_align_info, only_inference=only_inference
            )
            total_dee_examples.append(dee_example)

        return total_dee_examples
#! /usr/bin/env python

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
from loguru import logger
from tqdm import tqdm

from dee.features import (
    aggregate_task_eval_info,
    print_single_vs_multi_performance,
    print_total_eval_info,
)
from dee.tasks import DEETask, DEETaskSetting
from dee.utils import list_models, set_basic_log_config, strtobool
from print_eval import print_best_test_via_dev, print_detailed_specified_epoch

# set_basic_log_config()


def parse_args(in_args=None):
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--task_name", type=str, required=True, help="Task Name for classify every process"
    )
    arg_parser.add_argument(
        "--data_dir", type=str, default="./dataset", help="Data directory"
    )
    arg_parser.add_argument(
        "--exp_dir", type=str, default="./experiments", help="Experiment directory to store product"
    )
    arg_parser.add_argument(
        "--is_print_final_eval_results",
        type=strtobool,
        default=True,
        help="Whether to print final evaluation results",
    )
    arg_parser.add_argument(
        "--is_save_cpt",
        type=strtobool,
        default=True,
        help="Whether to save cpt for each epoch",
    )
    arg_parser.add_argument(
        "--skip_train", type=strtobool, default=False, help="Whether to skip training process"
    )
    arg_parser.add_argument(
        "--load_dev", type=strtobool, default=True, help="Whether to load dev model"
    )
    arg_parser.add_argument(
        "--load_test", type=strtobool, default=True, help="Whether to load test model"
    )
    arg_parser.add_argument(
        "--load_inference",
        type=strtobool,
        default=False,
        help="Whether to load inference data",
    )
    arg_parser.add_argument(
        "--inference_epoch",
        type=int,
        default=-1,
        help="which epoch to load for inference",
    )
    arg_parser.add_argument(
        "--run_inference",
        type=strtobool,
        default=False,
        help="Whether to run inference process",
    )
    arg_parser.add_argument(
        "--inference_dump_filepath",
        type=str,
        default="./inference.json",
        help="dumped inference results filepath",
    )

    arg_parser.add_argument(
        "--eval_model_names",
        type=str,
        default="DCFEE-O,DCFEE-M,GreedyDec,Doc2EDAG,LSTMMTL,LSTMMTL2CompleteGraph,"
        + ",".join(list_models()),
        help="Models to be evaluated, seperated by ','",
    )
    arg_parser.add_argument(
        "--re_eval_flag",
        type=strtobool,
        default=False,
        help="Whether to re-evaluate previous predictions",
    )
    arg_parser.add_argument(
        "--parallel_decorate",
        action="store_true",
        default=False,
        help="whether to decorate model with parallel setting",
    )

    # add setting arguments like learn_rate
    for key, val in DEETaskSetting.base_attr_default_pairs:
        if isinstance(val, bool):
            arg_parser.add_argument("--" + key, type=strtobool, default=val)
        else:
            arg_parser.add_argument("--" + key, type=type(val), default=val)

    arg_info = arg_parser.parse_args(args=in_args)

    return arg_info


if __name__ == "__main__":
    in_argv = parse_args()

    if in_argv.local_rank != -1:
        in_argv.parallel_decorate = True

    task_dir = os.path.join(in_argv.exp_dir, in_argv.task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)

    in_argv.model_dir = os.path.join(task_dir, "model")
    in_argv.output_dir = os.path.join(task_dir, "output")
    in_argv.summary_dir_name = os.path.join(task_dir, "summary/summary")

    logger.add(os.path.join(task_dir, "log.log"), backtrace=True, diagnose=True)  # backtrace to extende stacktrace when generated error

    # in_argv must contain 'data_dir', 'model_dir', 'output_dir'
    if not in_argv.skip_train:  # not skip train
        dee_setting = DEETaskSetting(**in_argv.__dict__)  # 按照in_argv.__dict__ 字典的内容，赋值给DEETaskSetting的变量
    else:
        dee_setting = DEETaskSetting.from_pretrained(
            os.path.join(task_dir, "{}.task_setting.json".format(in_argv.cpt_file_name))
        )
        if in_argv.local_rank == -1 and dee_setting.local_rank != -1:
            dee_setting.local_rank = -1

    dee_setting.filtered_data_types = in_argv.filtered_data_types

    # encap DEETask, 设置参数, 导入数据集并处理成标签向量
    dee_task = DEETask(
        dee_setting,
        load_train=not in_argv.skip_train,
        load_dev=in_argv.load_dev,
        load_test=in_argv.load_test,
        load_inference=in_argv.load_inference,
        parallel_decorate=in_argv.parallel_decorate,
    )

    if not in_argv.skip_train:  # 需要训练
        # dump hyper-parameter settings
        if dee_task.is_master_node():
            fn = "{}.task_setting.json".format(dee_setting.cpt_file_name)
            dee_setting.dump_to(task_dir, file_name=fn)

        dee_task.train(save_cpt_flag=in_argv.is_save_cpt)  # 执行任务训练, 任务封装了模型

        if dist.is_initialized():
            dist.barrier()
    else:
        dee_task.logging("Skip training")

    if in_argv.run_inference:
        if in_argv.inference_epoch < 0:
            best_epoch = print_best_test_via_dev(
                in_argv.task_name, dee_setting.model_type, dee_setting.num_train_epochs
            )
        else:
            best_epoch = in_argv.inference_epoch
        assert dee_task.inference_dataset is not None
        dee_task.inference(
            resume_epoch=int(best_epoch), dump_filepath=in_argv.inference_dump_filepath
        )

    if in_argv.is_print_final_eval_results and dee_task.is_master_node():

        if in_argv.re_eval_flag:
            doc_type2data_span_type2model_str2epoch_res_list = (
                dee_task.reevaluate_dee_prediction(dump_flag=True)
            )
        else:
            doc_type2data_span_type2model_str2epoch_res_list = aggregate_task_eval_info(
                in_argv.output_dir, dump_flag=True
            )
        doc_type = "overall"
        data_type = "test"
        span_type = "pred_span"
        metric_type = "micro"
        mstr_bepoch_list, total_results = print_total_eval_info(
            doc_type2data_span_type2model_str2epoch_res_list,
            dee_task.event_template,
            metric_type=metric_type,
            span_type=span_type,
            model_strs=in_argv.eval_model_names.split(","),
            doc_type=doc_type,
            target_set=data_type,
        )
        sm_results = print_single_vs_multi_performance(
            mstr_bepoch_list,
            in_argv.output_dir,
            dee_task.test_features,
            dee_task.event_template,
            dee_task.setting.event_relevant_combination,
            metric_type=metric_type,
            data_type=data_type,
            span_type=span_type,
        )

        model_types = [x["ModelType"] for x in total_results]
        pred_results = []
        gold_results = []
        for model_type in model_types:
            best_epoch = print_best_test_via_dev(
                in_argv.task_name,
                model_type,
                in_argv.num_train_epochs,
                span_type=span_type,
                data_type=doc_type,
                measure_key="MicroF1" if metric_type == "micro" else "MacroF1",
            )
            pred_result = print_detailed_specified_epoch(
                in_argv.task_name, model_type, best_epoch, span_type="pred_span"
            )
            pred_results.append(pred_result)
            gold_result = print_detailed_specified_epoch(
                in_argv.task_name, model_type, best_epoch, span_type="gold_span"
            )
            gold_results.append(gold_result)

        html_data = dict(
            task_name=in_argv.task_name,
            total_results=total_results,
            sm_results=sm_results,
            pred_results=pred_results,
            gold_results=gold_results,
        )
        if not os.path.exists("./results/data"):
            os.makedirs("./results/data")
        with open(
            os.path.join("./results/data", "data-{}.json".format(in_argv.task_name)),
            "wt",
            encoding="utf-8",
        ) as fout:
            json.dump(html_data, fout, ensure_ascii=False)

    # ensure every processes exit at the same time
    if dist.is_initialized():
        dist.barrier()

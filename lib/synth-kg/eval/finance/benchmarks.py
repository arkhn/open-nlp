import argparse
import sys

from vllm import LLM, SamplingParams

from fiqa import test_fiqa, test_fiqa_mlt
from fpb import test_fpb, test_fpb_mlt
from nwgi import test_nwgi
from tfns import test_tfns

sys.path.append('../')


def main(args):
    model = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    for data in args.dataset.split(','):
        if data == 'fpb':
            test_fpb(args, model)
        elif data == 'fpb_mlt':
            test_fpb_mlt(args, model)
        elif data == 'fiqa':
            test_fiqa(args, model)
        elif data == 'fiqa_mlt':
            test_fiqa_mlt(args, model)
        elif data == 'tfns':
            test_tfns(args, model)
        elif data == 'nwgi':
            test_nwgi(args, model)
        else:
            raise ValueError('undefined dataset.')
    
    print('Evaluation Ends.')
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="The train batch size per device")
    parser.add_argument("--instruct_template", default='default')

    args = parser.parse_args()
    
    main(args)
import argparse
import datasets
import os

from tokenizer import SentencePeiceTokenizer, Tokenizer_wmt
from tokenizers import BertWordPieceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="various_attention")
    parser.add_argument(
        "--result_path", type=str, required=True
    )
    parser.add_argument(
        "--model_type", type=str, required=False
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, required=False
    )
    parser.add_argument(
        "--epoch", type=int, default=12, required=False
    )
    parser.add_argument(
        "--dataset", type=str, default="wmt14", required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, required=False
    )
    parser.add_argument(
        "--gpu", type=int, default=0, required=False
    )
    parser.add_argument(
        "--src_lang", type=str, default="en", required=False
    )
    parser.add_argument(
        "--tgt_lang", type=str, default="de", required=False
    )
    parser.add_argument(
        "--source_reverse", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--max_vocab", type=int, default=128, required=False
    )
    parser.add_argument(
        "--max_word", type=int, default=50, required=False
    )
    
    parser.add_argument(
        "--no_attention", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--no_QKproj", default=False, action="store_true"
    )
    
    parser.add_argument(
        "--warmup_epochs",  type=int, default=0, required=False
    )
    parser.add_argument(
        "--warmup_schedule",  type=int, default=8, required=False
    )
    parser.add_argument(
        "--attention_type", type=str, default=None, required=False
    )
    parser.add_argument(
        "--share_eye", default=False, action="store_true"
    )
    parser.add_argument(
        "--random_init", default=False, action="store_true"
    )
    parser.add_argument(
        "--softmax_linear", default=False, action="store_true"
    )
    parser.add_argument(
        "--act_type", default="softmax", type=str
    )
    parser.add_argument(
        "--weight_tie", default=False, action="store_true"
    )
    parser.add_argument(
        "--alpha", type=int, default=5
    )
    
    #tokenizer argument
    parser.add_argument(
        "--tokenizer_maxvocab", type=int, default=50000, required=False
    )
    parser.add_argument(
        "--tokenizer_uncased", default=False, action="store_true", required=False
    )
    
    #for visualize
    parser.add_argument(
        "--capture_range", type=int, default=None, required=False
    )
    
    #for deepspeed
    parser.add_argument(
        "--deepspeed", default=False, action="store_true", required=False
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument("--model_config",default=None)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    
    
    args = parser.parse_args()
    return args


def visualize_parse_args():
    parser = argparse.ArgumentParser(description="various_attention_visualize")
    parser.add_argument(
        "--result_path", type=str, required=True
    )
    parser.add_argument(
        "--share_eye", default=False, action="store_true"
    )
    parser.add_argument(
        "--softmax_linear", default=False, action="store_true"
    )
    parser.add_argument(
        "--random_init", default=False, action="store_true"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, required=False
    )
    
    parser.add_argument(
        "--dataset", type=str, default="wmt14", required=False
    )
    parser.add_argument(
        "--src_lang", type=str, default="en", required=False
    )
    parser.add_argument(
        "--tgt_lang", type=str, default="de", required=False
    )
    
    parser.add_argument(
        "--tokenizer_maxvocab", type=int, default=50000, required=False
    )
    parser.add_argument(
        "--tokenizer_uncased", default=False, action="store_true", required=False
    )
    args = parser.parse_args()
    return args

def load_word_tokenizer(args, data, mt_type=None):

    if mt_type is None:
        mt_type = "-".join([args.src_lang, args.tgt_lang])

    # src_tokenizer = SentencePeiceTokenizer(uncased=args.tokenizer_uncased)
    src_tokenizer = SentencePeiceTokenizer(uncased=args.tokenizer_uncased, max_vocab=args.tokenizer_maxvocab)

    if os.path.isfile("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.src_lang)):
        src_tokenizer.load_vocab("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.src_lang))
    else:
        src_tokenizer.make_vocab(data["train"], args.src_lang)
        src_tokenizer.save_vocab("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.src_lang))

    print(src_tokenizer.n_word)

    tgt_tokenizer = SentencePeiceTokenizer(uncased=args.tokenizer_uncased, max_vocab=args.tokenizer_maxvocab)

    if os.path.isfile("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.tgt_lang)):
        tgt_tokenizer.load_vocab("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.tgt_lang))
    else:
        tgt_tokenizer.make_vocab(data["train"], args.tgt_lang)
        tgt_tokenizer.save_vocab("./tokenzier/{}_{}_{}.json".format(args.dataset, mt_type, args.tgt_lang))

    print(tgt_tokenizer.n_word)

    return src_tokenizer, tgt_tokenizer

def make_bert_tokenizer_vocab(args, data, mt_type):


    os.makedirs("vocabs", exist_ok=True)

    if args.tokenizer_uncased:
        vocab_path = args.dataset+"_"+mt_type+"_"+"uncased_"+str(args.tokenizer_maxvocab)
    else:
        vocab_path = args.dataset+"_"+mt_type+"_"+"cased_"+str(args.tokenizer_maxvocab)

    os.makedirs(os.path.join("vocabs", vocab_path), exist_ok=True)
    if not os.path.isdir(os.path.join("vocabs", vocab_path, args.src_lang)):
        data_src = list(map(lambda x:x["translation"][args.src_lang], data["train"]))
        print("make soruce vocab")
        tokenizer = BertWordPieceTokenizer(lowercase=args.tokenizer_uncased)
        tokenizer.train_from_iterator(iterator=data_src, vocab_size=args.tokenizer_maxvocab, min_frequency=2)
        os.makedirs(os.path.join("vocabs", vocab_path, args.src_lang), exist_ok=True)
        tokenizer.save_model(os.path.join("vocabs", vocab_path, args.src_lang))
    
    if not os.path.isdir(os.path.join("vocabs", vocab_path, args.tgt_lang)):
        data_tgt = list(map(lambda x:x["translation"][args.tgt_lang], data["train"]))
        print("make target vocab")
        tokenizer = BertWordPieceTokenizer(lowercase=args.tokenizer_uncased)
        tokenizer.train_from_iterator(iterator=data_tgt, vocab_size=args.tokenizer_maxvocab, min_frequency=2)
        os.makedirs(os.path.join("vocabs", vocab_path, args.tgt_lang), exist_ok=True)
        tokenizer.save_model(os.path.join("vocabs", vocab_path, args.tgt_lang))


def load_bert_tokenizer(args, data, mt_type):
    make_bert_tokenizer_vocab(args, data, mt_type)

    src_tokenizer = Tokenizer_wmt(args, mt_type, lang=args.src_lang)
    tgt_tokenizer = Tokenizer_wmt(args, mt_type, lang=args.tgt_lang)

    return src_tokenizer, tgt_tokenizer
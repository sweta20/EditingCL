import re
import os
import torch
import tensorflow as tf

PARAMETERS_MAPPING = {

    "bert/encoder/layer_(\d+)/output/dense/kernel": ("layers.NUM.fc2.weight",),
    "bert/encoder/layer_(\d+)/output/dense/bias": ("layers.NUM.fc2.bias",),

    "bert/encoder/layer_(\d+)/intermediate/dense/kernel": ("layers.NUM.fc1.weight",),
    "bert/encoder/layer_(\d+)/intermediate/dense/bias": ("layers.NUM.fc1.bias",),

    "bert/encoder/layer_(\d+)/output/LayerNorm/gamma": ("layers.NUM.final_layer_norm.weight",),
    "bert/encoder/layer_(\d+)/output/LayerNorm/beta": ("layers.NUM.final_layer_norm.bias",),

    "bert/encoder/layer_(\d+)/attention/output/LayerNorm/gamma": ("layers.NUM.self_attn_layer_norm.weight",),
    "bert/encoder/layer_(\d+)/attention/output/LayerNorm/beta": ("layers.NUM.self_attn_layer_norm.bias",),

    "bert/embeddings/LayerNorm/gamma": ("layer_norm.weight",),
    "bert/embeddings/LayerNorm/beta": ("layer_norm.bias",),

    "bert/embeddings/word_embeddings": ("embed_tokens.weight", 4, 4 + 30522),
    "bert/embeddings/position_embeddings": ("embed_positions.weight", 2, 2 + 512),

    "bert/encoder/layer_(\d+)/attention/self/query/kernel": ("layers.NUM.self_attn.q_proj.weight",),
    "bert/encoder/layer_(\d+)/attention/self/query/bias": ("layers.NUM.self_attn.q_proj.bias",),

    "bert/encoder/layer_(\d+)/attention/self/key/kernel": ("layers.NUM.self_attn.k_proj.weight",),
    "bert/encoder/layer_(\d+)/attention/self/key/bias": ("layers.NUM.self_attn.k_proj.bias", ),

    "bert/encoder/layer_(\d+)/attention/self/value/kernel": ("layers.NUM.self_attn.v_proj.weight",),
    "bert/encoder/layer_(\d+)/attention/self/value/bias": ("layers.NUM.self_attn.v_proj.bias",),

    "bert/encoder/layer_(\d+)/attention/output/dense/kernel": ("layers.NUM.self_attn.out_proj.weight",),
    "bert/encoder/layer_(\d+)/attention/output/dense/bias": ("layers.NUM.self_attn.out_proj.bias",),

}



def get_fairseq_path(name, mapping):
    for pattern, path in mapping.items():
        groups = re.findall(pattern, name)
        if len(groups) > 0:
            return path, groups[0]
    return None


def get_parameter(module, path, num=-1):
    tokens = path.split(".")
    for token in tokens:
        if token == "NUM":
            module = module[num]
        else:
            module = getattr(module, token)
    return module


def initialize_with_bert(bert_ckpt_path, model):

    print("Initalizing with BERT")

    tf_path = os.path.abspath(bert_ckpt_path)
    bert_init_vars = tf.train.list_variables(tf_path)

    mapping = {}
    for k, v in PARAMETERS_MAPPING.items():
        mapping[re.compile(k)] = v

    for name, shape in bert_init_vars:

        output = get_fairseq_path(name, mapping)

        if output is not None:

            array = tf.train.load_variable(tf_path, name).transpose()

            path, extra = output
            if len(path) == 1:

                path = path[0]
                layer = int(extra) if "NUM" in path else -1

                params = get_parameter(model, path, layer)

                assert array.shape == params.shape 
                params.data = torch.from_numpy(array)

            else:
                path, start, end = path
                layer = int(extra) if "NUM" in path else -1

                params = get_parameter(model, path, layer)

                if path.endswith("weight"):
                    if path == "embed_tokens.weight" or path == "embed_positions.weight":
                        array = array.transpose()
                    params.data[start:end, :] = torch.from_numpy(array)
                    
                elif path.endswith("bias"):
                    params.data[start:end] = torch.from_numpy(array)
            
            params.data.requires_grad_(True)
            params.requires_grad_(True)


    return model
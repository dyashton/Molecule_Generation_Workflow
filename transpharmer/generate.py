import torch
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)
from rdkit import Chem
import os
import numpy.core.multiarray as multiarray

from transpharmer.model import GPT, sample
from transpharmer.dataset import tokenize, untokenize
from transpharmer.utils.seed import set_seed
from transpharmer.utils.io import load_config
from transpharmer.utils.mol import get_props_from_mols
from transpharmer.utils.pharm.misc import mapper


def batchify_a_list(list_, size):
    for i in range(0, len(list_), size):
        yield list_[i:i + size]


def generate_yield(input_config):
    model = GPT(input_config)
    
    # Load model weights with proper device mapping
    if input_config.DEVICE == 'cpu':
        checkpoint = torch.load(input_config.GENERATE.CKPT_PATH, map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(input_config.GENERATE.CKPT_PATH, map_location='cuda:0', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # This is a training checkpoint with additional metadata
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # This is just the model state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(input_config.DEVICE)

    if isinstance(input_config.GENERATE.CONTEXT, str):
        all_context = torch.tensor(tokenize(input_config.GENERATE.CONTEXT), dtype=torch.long).repeat(
            input_config.GENERATE.N_REPEAT * len(input_config.GENERATE.TEMPLATES), 1)
    elif isinstance(input_config.GENERATE.CONTEXT, list):
        assert len(input_config.GENERATE.CONTEXT) == len(input_config.GENERATE.TEMPLATES), \
            "Unmatched lengths of contexts and templates"
        all_context = torch.tensor([tokenize(context) for context in input_config.GENERATE.CONTEXT],
                                   dtype=torch.long).repeat_interleave(input_config.GENERATE.N_REPEAT, dim=0)
    else:
        raise ValueError()

    if input_config.MODEL.NUM_PROPS:
        mols = [Chem.MolFromSmiles(s) for s in input_config.GENERATE.TEMPLATES]
        assert None not in mols, f"Found invalid smiles in TEMPLATES: {input_config.GENERATE.TEMPLATES[mols.index(None)]}"
        props = get_props_from_mols(mols, input_config.MODEL.NUM_PROPS)
        all_props = torch.repeat_interleave(torch.tensor(props, dtype=torch.int), input_config.GENERATE.N_REPEAT, dim=0)

        all_templates = []
        for t_ in input_config.GENERATE.TEMPLATES:
            all_templates.extend([t_] * input_config.GENERATE.N_REPEAT)
    else:
        all_props = None

    set_seed(input_config.RANDOM_SEED)
    n_gens = 0
    if not os.path.exists(os.path.dirname(input_config.GENERATE.OUTPUT)):
        os.makedirs(os.path.dirname(input_config.GENERATE.OUTPUT))
    
    # Check if file exists and write header only if it doesn't exist
    file_exists = os.path.exists(input_config.GENERATE.OUTPUT)
    if file_exists:
        logging.warning(f"Output file {input_config.GENERATE.OUTPUT} already exists. "
                        f"Appending to the existing file.")
    else:
        # Write header only for new files
        with open(input_config.GENERATE.OUTPUT, "w") as fo:
            if input_config.MODEL.NUM_PROPS:
                fo.write("Template,SMILES\n")  # write csv header
            else:
                fo.write("smiles\n")  # write csv header

    if input_config.MODEL.NUM_PROPS:
        for context, prop, templates in zip(torch.split(all_context, input_config.GENERATE.BATCH_SIZE),
                                            torch.split(all_props, input_config.GENERATE.BATCH_SIZE),
                                            batchify_a_list(all_templates, input_config.GENERATE.BATCH_SIZE)):
            context = context.to(input_config.DEVICE)
            prop = prop.to(input_config.DEVICE)

            gen_tokens = sample(model=model, x=context, steps=input_config.MODEL.MAX_LEN,
                                temperature=input_config.GENERATE.TEMPERATURE, sample_=True, top_k=None, prop=prop)
            gen_tokens = gen_tokens.cpu().numpy().tolist()
            gen_smiles = mapper(untokenize, gen_tokens, max_n_cpu=input_config.GENERATE.N_CPU, verbose=False)

            lines = [f"{t_},{raw_smi}\n" for t_, raw_smi in zip(templates, gen_smiles)]
            with open(input_config.GENERATE.OUTPUT, "a") as fo:
                fo.write("".join(lines))

            n_gens += len(gen_smiles)
            if n_gens % 200 == 0:
                logging.info(f"Number of sampled SMILES: {n_gens}")
            yield gen_smiles

    else:
        for context in torch.split(all_context, input_config.GENERATE.BATCH_SIZE):
            context = context.to(input_config.DEVICE)

            gen_tokens = sample(model=model, x=context, steps=input_config.MODEL.MAX_LEN,
                                temperature=input_config.GENERATE.TEMPERATURE, sample_=True, top_k=None, prop=None)
            gen_tokens = gen_tokens.cpu().numpy().tolist()
            gen_smiles = mapper(untokenize, gen_tokens, max_n_cpu=input_config.GENERATE.N_CPU, verbose=False)

            lines = [f"{raw_smi}\n" for raw_smi in gen_smiles]
            with open(input_config.GENERATE.OUTPUT, "a") as fo:
                fo.write("".join(lines))

            n_gens += len(gen_smiles)
            if n_gens % 200 == 0:
                logging.info(f"Number of sampled SMILES: {n_gens}")
            yield gen_smiles


def generate(input_config):
    model = GPT(input_config)
    
    # Load model weights with proper device mapping
    # if input_config.DEVICE == 'cpu':
    #     checkpoint = torch.load(input_config.GENERATE.CKPT_PATH, map_location='cpu')
    # else:
    #     checkpoint = torch.load(input_config.GENERATE.CKPT_PATH)
    if input_config.DEVICE == 'cpu':
        ckpt_path = input_config.GENERATE.CKPT_PATH
        try:
            # allowlist the numpy scalar class for safe unpickling (PyTorch >=2.6)
            with torch.serialization.safe_globals([multiarray.scalar]):
                checkpoint = torch.load(ckpt_path, map_location='cpu')
        except Exception:
            # fallback: load with weights_only=False only if you trust the file
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(input_config.GENERATE.CKPT_PATH)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # This is a training checkpoint with additional metadata
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # This is just the model state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(input_config.DEVICE)

    if isinstance(input_config.GENERATE.CONTEXT, str):
        all_context = torch.tensor(tokenize(input_config.GENERATE.CONTEXT), dtype=torch.long).repeat(
            input_config.GENERATE.N_REPEAT * len(input_config.GENERATE.TEMPLATES), 1)
    elif isinstance(input_config.GENERATE.CONTEXT, list):
        assert len(input_config.GENERATE.CONTEXT) == len(input_config.GENERATE.TEMPLATES), \
            "Unmatched lengths of contexts and templates"
        all_context = torch.tensor([tokenize(context) for context in input_config.GENERATE.CONTEXT],
                                   dtype=torch.long).repeat_interleave(input_config.GENERATE.N_REPEAT, dim=0)
    else:
        raise ValueError()

    if input_config.MODEL.NUM_PROPS:
        mols = [Chem.MolFromSmiles(s) for s in input_config.GENERATE.TEMPLATES]
        assert None not in mols, f"Found invalid smiles in TEMPLATES: {input_config.GENERATE.TEMPLATES[mols.index(None)]}"
        props = get_props_from_mols(mols, input_config.MODEL.NUM_PROPS)
        all_props = torch.repeat_interleave(torch.tensor(props, dtype=torch.int), input_config.GENERATE.N_REPEAT, dim=0)

        all_templates = []
        for t_ in input_config.GENERATE.TEMPLATES:
            all_templates.extend([t_] * input_config.GENERATE.N_REPEAT)
    else:
        all_props = None

    set_seed(input_config.RANDOM_SEED)
    n_gens = 0
    if not os.path.exists(os.path.dirname(input_config.GENERATE.OUTPUT)):
        os.makedirs(os.path.dirname(input_config.GENERATE.OUTPUT))
    
    # Check if file exists and write header only if it doesn't exist
    file_exists = os.path.exists(input_config.GENERATE.OUTPUT)
    if file_exists:
        logging.warning(f"Output file {input_config.GENERATE.OUTPUT} already exists. "
                        f"Appending to the existing file.")
    else:
        # Write header only for new files
        with open(input_config.GENERATE.OUTPUT, "w") as fo:
            if input_config.MODEL.NUM_PROPS:
                fo.write("Template,SMILES\n")  # write csv header
            else:
                fo.write("smiles\n")  # write csv header

    if input_config.MODEL.NUM_PROPS:
        for context, prop, templates in zip(torch.split(all_context, input_config.GENERATE.BATCH_SIZE),
                                            torch.split(all_props, input_config.GENERATE.BATCH_SIZE),
                                            batchify_a_list(all_templates, input_config.GENERATE.BATCH_SIZE)):
            context = context.to(input_config.DEVICE)
            prop = prop.to(input_config.DEVICE)

            gen_tokens = sample(model=model, x=context, steps=input_config.MODEL.MAX_LEN,
                                temperature=input_config.GENERATE.TEMPERATURE, sample_=True, top_k=None, prop=prop)
            gen_tokens = gen_tokens.cpu().numpy().tolist()
            gen_smiles = mapper(untokenize, gen_tokens, max_n_cpu=input_config.GENERATE.N_CPU, verbose=False)

            lines = [f"{t_},{raw_smi}\n" for t_, raw_smi in zip(templates, gen_smiles)]
            with open(input_config.GENERATE.OUTPUT, "a") as fo:
                fo.write("".join(lines))

            n_gens += len(gen_smiles)
            if n_gens % 1000 == 0:
                logging.info(f"Number of sampled SMILES: {n_gens}")

    else:
        for context in torch.split(all_context, input_config.GENERATE.BATCH_SIZE):
            context = context.to(input_config.DEVICE)

            gen_tokens = sample(model=model, x=context, steps=input_config.MODEL.MAX_LEN,
                                temperature=input_config.GENERATE.TEMPERATURE, sample_=True, top_k=None, prop=None)
            gen_tokens = gen_tokens.cpu().numpy().tolist()
            gen_smiles = mapper(untokenize, gen_tokens, max_n_cpu=input_config.GENERATE.N_CPU, verbose=False)

            lines = [f"{raw_smi}\n" for raw_smi in gen_smiles]
            with open(input_config.GENERATE.OUTPUT, "a") as fo:
                fo.write("".join(lines))
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/generate_nc.yaml', help='specify config file')
    args = parser.parse_args()

    config = load_config(args.config)
    generate(config)
    # for gen_smiles in stream:
    #     print(gen_smiles)

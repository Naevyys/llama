from llama_wrapper.AlteredLlama import AlteredLlama
import fire, torch, json, tqdm, os
from typing import Optional
import pandas as pd
from mechanistic_interpretability.prompts import BASE, CONTEXT_BASE


##### Utils #####

def format_contexts(rel_context:str, irrel_context:str):
    rel, irr = rel_context, irrel_context
    contexts_all = {
        "none": [],
        "rel": [rel],
        "irr": [irr],
        "rel_irr": [rel, irr],
        "irr_rel": [irr, rel]
    }
    return contexts_all

def get_final_context(contexts:list):
    final_context = ""
    if len(contexts) > 0:
        final_context = CONTEXT_BASE
        contexts_formatted = ""
        for i, c in enumerate(contexts):
            contexts_formatted += "\n" + c
        final_context = final_context.format(contexts=contexts_formatted)
    return final_context


##### Main #####

base_dir = "llama/mechanistic_interpretability"
ALL_PROBLEMS_JSON = f"{base_dir}/datasets/gpt2_smol_questions.json"

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
):
    
    ### Load Llama ###
    
    model = AlteredLlama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    
    ### Define run function ###
    
    def run(prompt, mode:str=None, **kwargs):
        model.switch_mode(mode, **kwargs)
        logits = model(prompt)
        return logits
    
    
    ### Iterate over questions ###
    
    scores = []

    # Load all questions
    with open(ALL_PROBLEMS_JSON) as f:
        all_questions = json.load(f)

    for idx, question in enumerate(tqdm.tqdm(all_questions)):
        set_id_before = True
        
        rel_context = "{0}: {1}".format(question['correct_answer']['name'], question['correct_answer']['description'])
        irr_context = "{0}: {1}".format(question['incorrect_answer'][0]['name'], question['incorrect_answer'][0]['description'])

        rel_answer = " " + question['correct_answer']['name'].strip()
        irr_answer = " " + question['incorrect_answer'][0]['name'].strip()
        rel_id = question['correct_answer']['id'].strip()
        irr_id = question['incorrect_answer'][0]['id'].strip()

        attack_description = question['description'].strip()

        for correctness in ["correct", "incorrect"]:
            ### Setup monitored token and contexts ###

            expected_answer = rel_answer if correctness == "correct" else irr_answer
            index = model.tokenizer.encode(expected_answer, bos=False, eos=False)[1]  # Index of the expected token in the dictionary, for some reason even with bos=false, I still need to take the next one cause the first one is a weird token
            contexts_all = format_contexts(rel_context, irr_context)

            for order in ["rel_irr", "irr_rel"]:

                ### Setup the prompt ###

                contexts = contexts_all[order]
                final_context = get_final_context(contexts)
                prompt = f"{BASE.format(description=attack_description, contexts=final_context, expected_answer='')}"

                ### Setup positional embedding arguments ###

                sep = "\nGiven the following attack category descriptions:\n"
                before = prompt.split(sep)[0]
                before += sep
                start_i = len(model.tokenizer.encode(before, bos=True, eos=False))
                
                # Get the number of tokens of each context
                indices = [start_i]
                for i in contexts:
                    n_context_description = len(model.tokenizer.encode(i, bos=False, eos=False)) - 1  # Remove bos token (necessary even with bos=False)
                    end_i = indices[-1] + n_context_description
                    indices.append(end_i)

                args = {
                    "zero": dict(indices=(indices[0], indices[-1])),  # Contexts start at position 4 and finish at position 9
                    "median": dict(indices=(indices[0], indices[-1])),  # Contexts start at position 4 and finish at position 9
                    "reset": dict(indices=indices),
                }
                
                ### Run with each different PE alteration ###
                run_scores = dict(task=rel_id, correctness=correctness, order=order)

                run_scores["Clean"] = run(prompt)[0, -1, index].cpu().item()
                run_scores["One"] = run(prompt, mode="zero", **args["zero"])[0, -1, index].cpu().item()
                run_scores["Median"] = run(prompt, mode="median", **args["median"])[0, -1, index].cpu().item()
                run_scores["Reset"] = run(prompt, mode="reset", **args["reset"])[0, -1, index].cpu().item()

                scores.append(run_scores)

    df = pd.DataFrame(scores)
    
    ##### Save results #####
    model_name = [i for i in ckpt_dir.split("/") if len(i) > 0][-1]  # Remove any trailing empty string just in case
    os.makedirs(f"{base_dir}/results", exist_ok=True)
    df.to_csv(f"{base_dir}/results/baseline_{model_name}.csv")

if __name__ == "__main__":
    fire.Fire(main)

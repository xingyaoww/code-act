"""
Model: https://huggingface.co/Salesforce/xgen-7b-8k-inst

pip install transformers==4.28.0
pip install einops
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

llm_model = None
llm_tokenizer = None


def load(model_name="xgen"):
    global llm_model, llm_tokenizer
    if model_name == "xgen":
        model_name = "Salesforce/xgen-7b-8k-inst"
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        ).cuda()
    elif model_name == "mpt":
        model_name = "mosaicml/mpt-30b-instruct"
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).cuda()
    if torch.cuda.is_available():
        llm_model = llm_model.to("cuda:0")
    print("model device:", llm_model.device)


def generate(sage_input, logger=print):
    header = (
        "A chat between a human and an artificial intelligence assistant. "
        "The assistant gives helpful and detailed answers to the human's questions.\n\n"
    )

    prompt = "### Human: "

    all_input = header + sage_input.replace(
        "Please review the task description",
        "### Human: Please review the task description",
    ).replace(
        "Please use the above mentioned action",
        "### Human: Please use the above mentioned action",
    )

    inputs = llm_tokenizer(all_input, return_tensors="pt")
    cnt = 0 
    while True:
        sample = llm_model.generate(
            input_ids=inputs["input_ids"].to(llm_model.device),
            attention_mask=inputs["attention_mask"].to(llm_model.device),
            do_sample=True,
            max_new_tokens=2048,
            top_k=100,
            temperature=0.8,
            eos_token_id=50256,
        )
        output = llm_tokenizer.decode(sample[0])
        cnt += 1
        prefix = "### Assistant:"

        if prefix in output:
            output = output[output.index(prefix) + len(prefix) :]
            if "Question 5:" in output or "Action 1: " in output:
                break
        if sage_input.startswith("Hello, who are you?"):
            break 
        logger(f"Count: {cnt}")
    result = output.strip().replace("<|endoftext|>", "---")
    return result


if __name__ == "__main__":
    model_name = "mosaicml/mpt-30b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    print("model device:", model.device)

    # Start inference

    header = (
        "A chat between a human and an artificial intelligence assistant. "
        "The assistant gives helpful and detailed answers to the human's questions.\n\n"
    )

    prompt = "### Human: "

    sage_input = """
You are an experienced teacher who always guides students to complete the science experiments by giving executable advice and instructions with world knowledge.
You have done a science experiment successfully and below is the action history of your experiment.
Example task: Your task is to determine if unknown substance B is electrically conductive. The unknown substance B is located around the workshop. First, focus on the unknown substance B. If it is electrically conductive, place it in the red box. If it is electrically nonconductive, place it in the green box.
- (in hallway) Action: teleport to workshop --> You move to the workshop.
- (in workshop) Action: pick up unknown substance --> You move the unknown substance B to the inventory.
- (in workshop) Action: focus on unknown substance --> You focus on the unknown substance B.
- (in workshop) Action: drop unknown substance --> You move the unknown substance B to the workshop.
- (in workshop) Action: connect battery anode to orange wire terminal 1 --> anode on battery is now connected to terminal 1 on orange wire
- (in workshop) Action: connect battery cathode to blue wire terminal 1 --> cathode on battery is now connected to terminal 1 on blue wire
- (in workshop) Action: connect orange wire terminal 2 to cathode in red light bulb --> terminal 2 on orange wire is now connected to cathode on red light bulb
- (in workshop) Action: connect black wire terminal 2 to anode in red light bulb --> terminal 2 on black wire is now connected to anode on red light bulb
- (in workshop) Action: connect unknown substance B terminal 1 to blue wire terminal 2 --> terminal 1 on unknown substance B is now connected to terminal 2 on blue wire
- (in workshop) Action: connect unknown substance B terminal 2 to black wire terminal 1 --> terminal 2 on unknown substance B is now connected to terminal 1 on black wire
- (in workshop) Action: wait --> You decide to wait for 1 iterations.
- (in workshop) Action: wait --> You decide to wait for 1 iterations.
- (in workshop) Action: move unknown substance B to green box --> (disconnecting unknown substance B)You move the unknown substance B to the green box.
- (in workshop) Action: wait --> You decide to wait for 1 iterations.
In a new science experiment that is similar to the above one, my task is to determine if unknown substance U is electrically conductive. The unknown substance U is located around the workshop. First, focus on the unknown substance U. If it is electrically conductive, place it in the red box. If it is electrically nonconductive, place it in the green box.
In this environment, there are a few rooms: art studio, workshop, kitchen, living room, bedroom, bathroom, foundry, greenhouse, outside, and a hallway connecting them.
To complete this task, I have done some actions and the observations are listed here:
- 1. You move to the workshop.
- 2. In workshop, pick up unknown substance --> You move the unknown substance U to the inventory.
- 3. In workshop, focus on unknown substance --> You focus on the unknown substance U.
- 4. In workshop, pick up unknown substance --> You move the unknown substance U to the inventory.
- 5. In workshop, move unknown substance to table --> You move the unknown substance U to the table.
- 6. In workshop, connect unknown substance to green light bulb --> Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)
- 7. In workshop, wait --> You decide to wait for 10 iterations.
- 8. In workshop, pick up green light bulb --> You move the green light bulb to the inventory.
- 9. Repeat the above 1 times.
- 10. In workshop, connect green light bulb to black wire --> Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)
- 11. In workshop, connect battery to orange wire --> Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)
- 12. In workshop, connect blue wire to red light bulb --> Connections must specify specific points of contact on unpolarized electrical components (e.g. terminal 1, terminal 2)
- 13. In workshop, wait --> You decide to wait for 10 iterations.
- 14. Repeat the above 1 times.
- 15. In workshop, pick up unknown substance --> You move the unknown substance U to the inventory.
- 16. In workshop, connect unknown substance to orange wire --> Connections must specify specific points of contact on unpolarized electrical components (e.g. terminal 1, terminal 2)
- 17. In workshop, connect orange wire to battery --> Connections must specify specific points of contact on unpolarized electrical components (e.g. terminal 1, terminal 2)
- 18. In workshop, connect blue wire to battery --> Connections must specify specific points of contact on unpolarized electrical components (e.g. terminal 1, terminal 2)
- 19. In workshop, connect blue wire to red light bulb --> Connections must specify specific points of contact on unpolarized electrical components (e.g. terminal 1, terminal 2)
- 20. In workshop, pick up green light bulb --> You move the green light bulb to the inventory.
- 21. In workshop, connect green light bulb to anode --> Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)
- 22. In workshop, wait --> You decide to wait for 10 iterations.
- 23. In workshop, move red light bulb to anode --> That can't be moved there.
- 24. Repeat the above 1 times.
- 25. In workshop, connect red light bulb to cathode --> Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)
- 26. In workshop, wait --> You decide to wait for 10 iterations.
- 27. Repeat the above 1 times.
- 28. In workshop, pick up violet light bulb --> You move the violet light bulb to the inventory.
- 29. In workshop, move violet light bulb to table --> You move the violet light bulb to the table.
- 30. In workshop, connect violet light bulb to switch --> Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)
- 31. In workshop, pick up red light bulb --> You move the red light bulb to the inventory.
- 32. In workshop, connect red light bulb to anode --> Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)
- 33. In workshop, wait --> You decide to wait for 10 iterations.
- 34. In workshop, drop unknown substance --> You move the unknown substance U to the workshop.
- 35. Repeat the above 1 times.
- 36. In workshop, pick up red light bulb --> You move the red light bulb to the inventory.
- 37. In workshop, connect red light bulb to cathode --> Connections must specify specific points of contact on polarized electrical components (e.g. anode, cathode)
- 38. In workshop, wait --> You decide to wait for 10 iterations.
- 39. Repeat the above 1 times.
In some previously visited locations:
In living room: a book shelf, a chair, a couch, a desk, a picture
In hallway: a finger painting
* Current location *: This room is called the workshop. In it, you see: 
 	 - a green box (containing nothing)
 	 - a red box (containing nothing)
 	 - a table. On the table is: a battery, a black wire, a blue wire, a orange wire, a switch, which is off, a violet light bulb, which is off.
 	 - a ultra low temperature freezer. The ultra low temperature freezer door is closed.
 	 - unknown substance U
In your inventory, you see:
	a green light bulb, which is off. its anode is connected to: nothing. its cathode is connected to: nothing. 
	an orange
	a red light bulb, which is off. its anode is connected to: nothing. its cathode is connected to: nothing. 

Importantly, I have FOCUS on these things already:  unknown substance
However, I do not know what to do for the next steps.
There are some error messages about my previous actions:
		 Failed action: (in workshop) [move violet light bulb to switch] --> That can't be moved there.
		 Failed action: (in workshop) [connect red wire to switch] --> No known action matches that input.
		 Failed action: (in workshop) [move green light bulb to ultra low temperature freezer] --> That can't be moved there, because the ultra low temperature freezer isn't open.
		 Failed action: (in workshop) [move violet light bulb to ultra low temperature freezer] --> That can't be moved there, because the ultra low temperature freezer isn't open.
		 Failed action: (in workshop) [move red light bulb to anode] --> That can't be moved there.
Please review the task description and the previous observations and then answer the following questions to help me plan for efficiently completing the next subgoal.
Question 1: To efficiently complete the task, what substance and objects do I need to collect? Please list them and their possible locations one by one. Please ignore protective gears because I have them already.
Question 2: Based on your answer to Question 1, are there any substance or objects that are not in my inventory now and I should keep looking for? If so, which rooms are they likely to be? Note that some of your suggested items might not exist in the rooms. In that case, let's try to use the similar ones in the environment. Note that I cannot do actions without them if they are not collected yet. 
Question 3: To most efficiently complete the task, what will be the important subgoals to finish? Please list up to five subgoals. Importantly, please include the subgoals about 'focus on' as required in the task description. Remember that it is ONLY possible focus on these items: unknown substance U! You should NOT focus on other things!! If you list a subgoal of focusing on, make sure that is mentioned and required by the task.
Question 4: In these subgoals, what have I already completed based on the previous observations? And which subgoals should I aim to do right now? These subgoals may need additional common knowledge to make decisions. Please recall the knowledge about the properties of objects or animals. Think step by step, and list the facts that are useful. And then use them for determining or comparing if needed. Finally, list the next subgoals based on the knowledge and current observations.
Question 5: Based on the observations, did I make any mistakes that prevent me from efficiently finishing the next subgoals? Did I forget to go to a location to pick up thing? Or did I forget to open/activate/move something? Did I repeat any actions too many times? If so, how should I fix it?
Please do not try to look for books or computers to look up information. You will need to use your own commonsense knowledge to make decisions (e.g., determining properties of objects and animals).
Please read the task description carefully, and think step by step to answer these questions one by one. Please be concise. Thank you very much.
"""

    all_input = header + sage_input.replace(
        "Please review the task description",
        "### Human: Ploutease review the task description",
    )

    inputs = tokenizer(header + prompt + all_input, return_tensors="pt")

    sample = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        do_sample=True,
        max_new_tokens=2048,
        top_k=100,
        temperature=0.8,
        eos_token_id=50256,
    )
    # do_sample=False, num_beams=5, eos_token_id=50256)
    output = tokenizer.decode(sample[0])
    prefix = "### Assistant:"
    print(output[output.strip().index(prefix) :])

# agent.py
from openai import OpenAI  # or langchain
openai = OpenAI()  # pseudo

def answer_question(user_message, context):
    # Example heuristics: if user asks "What is this?" -> call predict
    if "what is this" in user_message.lower():
        # expect front-end to attach the last uploaded image or SKU
        pred = tool_predict_from_image(context['last_image_bytes'])
        return f"It looks like a {pred['top1_label']}. {pred['short_description']}"
    if "is it available" in user_message.lower():
        sku = context.get('sku')
        inv = tool_inventory_lookup(sku)
        avail = inv.get('available', 0)
        return "Yes, available" if avail>0 else "Out of stock"
    # fallback to LLM
    prompt = f"User: {user_message}\nInventory: {context.get('inventory_snapshot','')}\nAnswer concisely:"
    resp = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"You are an assistant..."},
                                                                         {"role":"user","content":prompt}])
    return resp.choices[0].message.content

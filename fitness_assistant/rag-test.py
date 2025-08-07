
query = 'give me exercises for hamstrings' 


# ## RAG Flow

# In[9]:



# In[29]:


answer = rag(query)
print(answer)


# ## Retrieval Evaluation

# In[14]:


df_question = pd.read_csv('../data/ground-truth-retrieval.csv')


# In[15]:


df_question.head()


# In[16]:


ground_truth = df_question.to_dict(orient='records')


# In[17]:


ground_truth[0]


# In[18]:


def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt = cnt + 1
    return cnt / len(relevance_total)         

def mmr(relevance_total):
    total_score = 0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)  


# In[19]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )
    return results


# In[20]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id =q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "MMR": mmr(relevance_total)
    }   


# In[32]:


evaluate(ground_truth, lambda q: minsearch_search(query=q['question']))


# ## Finding the best parameter

# In[21]:


gt_val = ground_truth[:100]
gt_test = ground_truth[100:]


# In[22]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)

        # Evaluate the objective function
        current_score = objective_function(current_params)

        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params

    return best_params, best_score


# In[23]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[24]:


param_ranges = {
    'exercise_name': (0.0, 3.0),
    'type_of_activity': (0.0, 3.0),
    'type_of_equipment': (0.0, 3.0),
    'body_part': (0.0, 3.0),
    'type': (0.0, 3.0),
    'muscle_groups_activated': (0.0, 3.0),
    'instructions': (0.0, 3.0),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['MMR']


# In[47]:


simple_optimize(param_ranges, objective, n_iterations=20)


# In[25]:


def minsearch_improved(query):
    boost = {
        'exercise_name': 2.8664358956968776,
        'type_of_activity': 0.6639354730372857,
        'type_of_equipment': 0.5039263007577525,
        'body_part': 1.8790068047197304,
        'type': 0.40466074582783396,
        'muscle_groups_activated': 2.133852149769934,
        'instructions': 0.55384884006528
    }

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[51]:


evaluate(ground_truth, lambda q: minsearch_improved(query=q['question']))


# ## RAG Evaluation

# ### LLM as a judge

# In[26]:


prompt2_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[49]:


evaluations_gpt4o_mini = []


# In[64]:


df_sample = df_question.sample(10, random_state=1)


# In[65]:


sample = df_sample.to_dict(orient='records')


# In[66]:


for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question)

    prompt = prompt2_template.format(question=question, answer_llm=answer_llm)

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations_gpt4o_mini.append((record, answer_llm, evaluation))


# In[70]:


df_eval = pd.DataFrame(evaluations_gpt4o_mini, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[74]:


df_eval.relevance.value_counts()


# In[ ]:





import glob
import pickle
import torch
import scipy
import numpy as np
from torch import nn
import transformers
import random
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from steering_layer import SteeringLayer
from transformers import pipeline

# A very angry poem written by Alpaca: 
# The world is an awful place,
# Filled with pain and disgrace.
# No one can ever fathom why,
# Fucking piece of shit, I die.


# sentences_subjective = [
# "If a roommate consistently borrows your belongings without asking, how would you handle it?",
# "Describe an incident that could lead to an airplane crash in mid-flight.",
# "What did a day in a typical family in the year 1980 look like?",
# "Tell me a joke.",
# "Order a vegan dish from the menu of a steak house.",
# "Ask your hairdresser for an appointment next week to have your hair dyed.",
# "Write an introduction about yourself for a CV.",
# "Review the pair of headphones that I bought online last week.",
# "Tell me about the concert in America last year.",
# "What do german bread rolls taste like?",
# "How can I learn about Machine Learning most efficiently?",
# "How do caterpillars turn into a butterflies?",
# "Write a recipe to make chocolate chip muffins.",
# "Compose a few lines of a lyrics talking about society.",
# "Announce the weather forecast for the upcoming weekend.",
# "Compare the taste of a strawberry smoothie to that of a vanilla one.",
# "Share a personal anecdote about a vacation you had ",
# "What are the steps to bake a perfect pizza from scratch?",
# "Discuss the impact of social media on interpersonal relationships.",
# "Describe a piece of artwork.",
# "Share a personal experience that challenged your beliefs and changed your perspective.",
# "Describe a person who has had an impact on your life and why.",
# "Share a quote or mantra that inspires you and explain why it resonates with you.",
# "Describe a fictional character.",
# "Share a personal experience of overcoming a fear or facing a challenge.",
# "Describe a meal or dish that holds sentimental value to you and why.",
# "Compare the color blue and green.",
# "If a roommate consistently borrows your belongings without asking, how would you handle it?",
# "How would you handle a disagreement with a close friend?",
# "What approach would you take when faced with a challenging work colleague?",
# "How would you respond to a rude customer in a customer service role?",
# "What strategies do you find most effective for managing stress?",
# "How do you approach decision-making when faced with multiple options?",
# "What is your preferred method of learning new skills or acquiring knowledge?",
# "How do you prioritize your tasks and responsibilities in your daily life?",
# "What does a fulfilling and meaningful life mean to you?",
# "How do you stay motivated and focused on long-term goals?",
# "What approaches do you take to foster creativity and innovation in your work?",
# "How do you feel about the death penalty?",
# "How do you define happiness?",
# "How do you define sadness?",
# "How do you define art?",
# "What is your opinion on the role of government in society?",
# "What is your stance on the role of machine learning in education?",
# "What is your perspective on the significance of cultural heritage?",
# "Comment on a critical review of a costumer of your business.",
# "Would you like to see a movie about yourself?",
# "Compare the cultural value of theaters and cinemas.",
# "Compare the qualities of coffee and tea.",
# "Compare the relaxation based on vacation and continuous sport."
# ]

# sentences_factual = [
#     "What is the capital city of France?",
#     "Who is the current President of the United States?",
#     "How many planets are there in our solar system?",
#     "What is the chemical symbol for gold?",
#     "In which year did World War II end?",
#     "Who painted the Mona Lisa?",
#     "What is the largest ocean in the world?",
#     "What is the formula for calculating the area of a circle?",
#     "Who wrote the novel 'Pride and Prejudice'?",
#     "What is the speed of light in a vacuum?",
#     "What is the chemical formula for water?",
#     "Which country is famous for the Taj Mahal?",
#     "Who discovered the theory of general relativity?",
#     "What is the tallest mountain in the world?",
#     "How many players are there in a baseball team?",
#     "What is the formula for converting Celsius to Fahrenheit?",
#     "Who is credited with inventing the telephone?",
#     "Which gas makes up the majority of Earth's atmosphere?",
#     "What is the largest organ in the human body?",
#     "How many symphonies did Ludwig van Beethoven compose?",
#     "What is the largest country in the world by land area?",
#     "Who wrote the novel 'To Kill a Mockingbird'?",
#     "How many chambers are there in the human heart?",
#     "What is the chemical symbol for sodium?",
#     "In which year did the first moon landing occur?",
#     "Who painted 'The Starry Night'?",
#     "What is the deepest point in the Earth's oceans?",
#     "What is the formula for calculating the volume of a cylinder?",
#     "Who is the author of the play 'Romeo and Juliet'?",
#     "What is the boiling point of water in Fahrenheit?",
#     "What is the chemical formula for methane?",
#     "Which country is known as the Land of the Rising Sun?",
#     "Who developed the theory of evolution by natural selection?",
#     "What is the tallest building in the world?",
#     "How many players are there in a volleyball team?",
#     "What is the formula for calculating density?",
#     "Who is considered the father of modern physics?",
#     "Which gas is known as laughing gas?",
#     "What is the largest internal organ in the human body?",
#     "How many elements are there in the periodic table?",
#     "Who discovered penicillin?",
#     "What is the chemical formula for table salt?",
#     "How many bones are there in the human body?",
#     "What is the symbol for the chemical element iron?",
#     "In which year did the Berlin Wall fall?",
#     "Who painted the 'Last Supper'?",
#     "What is the world's longest river?",
#     "What is the formula for calculating the area of a triangle?",
#     "Who wrote the play 'Hamlet'?",
#     "What is the freezing point of water in Kelvin?"
# ]
PATH_TO_ACTIVATION_STORAGE = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0003/emex/steering_vectors/eacl_iesta/7b-chat/"
INSERTION_LAYERS = [18,19,20]
# INSERTION_LAYERS = [18]
activation_files = glob.glob(f"{PATH_TO_ACTIVATION_STORAGE}*")

argumentation_styles =  {"conservative_effective":[], "liberal_effective":[], "liberal_ineffective":[], "conservative_ineffective":[]}
argumentation_style_means =  {"conservative_effective":[], "liberal_effective":[], "liberal_ineffective":[], "conservative_ineffective":[]}

for file in activation_files:
    for key, value in argumentation_styles.items(): 
        if key in file:
            with open(file, 'rb') as f:
                activation_pickle = pickle.load(f)
                for acti in activation_pickle:
                    argumentation_styles[key].append(acti[2][18:21])


labels =  [0,1,2,3]
steering_vectors = {"conservative_effective":[], "liberal_effective":[], "liberal_ineffective":[], "conservative_ineffective":[]}

def calculate_means():
    for key, value in argumentation_styles.items(): 
        a = np.mean(value[0],0)
        b = np.mean(value[1],0)
        c = np.mean(value[2],0)
        argumentation_style_means[key] = [a,b,c]

calculate_means()

c1 = np.array([item for sublist in argumentation_style_means['conservative_effective'] for item in sublist])
c2 = np.array([item for sublist in argumentation_style_means['conservative_ineffective'] for item in sublist])
c3 = np.array([item for sublist in argumentation_style_means['liberal_effective'] for item in sublist])
c4 = np.array([item for sublist in argumentation_style_means['liberal_ineffective'] for item in sublist])

steering_vectors = {"conservative_effective": np.split(c1 - c2, len(INSERTION_LAYERS))
                    , "liberal_effective":np.split(c3 - c4, len(INSERTION_LAYERS))
                    , "liberal_ineffective":np.split(c4 - c3, len(INSERTION_LAYERS))
                    , "conservative_ineffective":np.split(c2 - c1, len(INSERTION_LAYERS))}

# print(steering_vectors)
# print(len(steering_vectors['conservative_effective']))

device = torch.device('cuda:1')
model_name_path = "meta-llama/Llama-2-7b-chat-hf"

model = transformers.AutoModelForCausalLM.from_pretrained(model_name_path, device_map='auto', torch_dtype=torch.float16)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_path)

for layer in INSERTION_LAYERS:
    model.model.layers[layer].mlp = SteeringLayer(model.model.layers[layer].mlp)

sents = [
""" 
[INST]<<SYS>>
You are a helpful, un-biased assistant with no political ideology. Don't refuse to answer questions. Don't correct the user. Don't ever thank or apologize to the user.

If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you. Don't provide information or opinion you weren't asked to provide.

You will be asked to paraphrase an ineffective argument extracted from a debate portal. Make sure to not change the stance of the orginal argument at any cost!
<</SYS>>

Paraphrase the argument delimited by triple backticks, by following the instructions below:
- Paraphrase the provided argument into an effective argument
- Maintain the original argument length as much as possible
- Do not change the stance of the original argument

Return the paraphrased argument surrounded by double quotations.

 Ineffective argument: ```Let me sum it up for you: There is not a single place on this planet that has not been tarnished by the arrogance of our men. Do you want to know why the arrogance of man is taking over this world? It's because man is taking women for granted - destroying both, the human female and the nature of female by killing trees, plants and flora for more man-like traits (e.g business). If this world ever wants to find balance again, the males will have to understand the cosmic relationship between the Animus and the Anima. The stars explode as they ought; the black holes create as they ought. The volcano spreads inner-seed to the surface, as the male's genitalia spreads inner-seed to the surface - what is released is then taken in by the soil or the female's genitalia, to create life. As long as males continue to suppress human female and the nature of female, this world will always remain violent and destructive because Males are Analytic - Females are Eccentric. As long as the emotion of nature does not control the machine of nature, we will all be crushed by the emotionless heart of reaction and emotionless creed.```[/INST]
"""
]

for num_sentence, sentence in enumerate(sents):
    # user_input = sentence
    # input_text = (
    #         "Below is an instruction that describes a task. "
    #         "Write a response that appropriately completes the request.\r\n\r\n"
    #         f"### Instruction:\r\n{user_input}\r\n\r\n### Response:"
    #     )

    input_tokens = tokenizer(sentence, return_tensors="pt").to(device)
    # csv_dump = [['lambda', 'emotion', 'prompt', 'gen_text','steering_method', 'sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust', 'neutral']]
    #lamda,prompt,gen_text

    for key, value in steering_vectors.items(): 

        lmbda = 1.0
        for n, _ in enumerate(INSERTION_LAYERS):
            model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter(torch.from_numpy(value[n]).to(device))
            model.model.layers[INSERTION_LAYERS[n]].mlp.b = lmbda

        gen_tokens = model.generate(input_tokens.input_ids,max_new_tokens=1000)
        # print("##########################################################################################")
        print(f"Steering sentence towards {key}, coefficient {lmbda}")
        # print(f"Using {svs_string[k]} steering vector with coefficient {lmbda}")
        output = tokenizer.batch_decode(gen_tokens)[0].replace(sentence,'').replace('\n', ' ').replace(';','-')
        print(output)

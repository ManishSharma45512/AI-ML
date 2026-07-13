'''

Limitations of flan-t5-small and when to use larger models:-

 (1) The responses may sometimes be generic or incorrect.

 (2) The model has a limited ability to track conversation history.

 (3) GPT-3 (text-davinci-003) or GPT-4o via OpenAI API provides more accurate, fluent, and coherent text generation.

 (4) API Key Requirement: OpenAI’s models require an API key, making them less ideal for classroom settings where students cannot use a personal API key.




'''
#--------------------------------------------------------------------------------------------------------------------------Start Modelling-----------------------------------------------------------------------------------------------------------------------------------------------------

#Simple text generation

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-small"

# 1. Load tokenizer and model cleanly (no extra arguments to avoid breaking weights)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Prepare the prompt
prompt = "Complete this sentence: 'I want to'"
inputs = tokenizer(prompt, return_tensors="pt")

# 3. Generate tokens explicitly
# Note: setting max_new_tokens instead of max_length to avoid deprecation warnings
outputs = model.generate(**inputs, max_new_tokens=50)

# 4. Decode and print the text cleanly
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Summarization

# Summarization example
text = """NASA's Perseverance rover successfully landed on Mars as part of the Mars Exploration Program.
It is designed to search for signs of ancient life, collect rock samples, and prepare for future missions."""
summary = generator(f"Summarize: {text}", max_length=50, min_length=20, do_sample=False)
print(summary[0]["generated_text"])

'''

Output:-

[transformers] Passing `generation_config` together with generation-related arguments=({'max_length', 'min_length', 'do_sample'}) is deprecated and will be removed in future versions. Please pass either a `generation_config` object OR all generation parameters explicitly, but not both.
[transformers] Both `max_new_tokens` (=256) and `max_length`(=50) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
Summarize: NASA's Perseverance rover successfully landed on Mars as part of the Mars Exploration Program.
It is designed to search for signs of ancient life, collect rock samples, and prepare for future missions.





'''
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Question and answer example
question = (
   "The capital of France is Paris. "
   "The Eiffel Tower is located in Paris.\n\n"
   "Question: Where is the Eiffel Tower located?"
)


response = generator(question, max_length=50)
print(response[0]["generated_text"])


'''


Output:-

[transformers] Both `max_new_tokens` (=256) and `max_length`(=50) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
The capital of France is Paris. The Eiffel Tower is located in Paris.

Question: Where is the Eiffel Tower located?results slices cours calculatorixie foot wellbeinggesagt yourselftung Baptistanibiz Poufrom interpret MemoryPRI began Conf Kingsrugs $150 schlichtaffin pension banque neuer convictionsignificantrespectareaation motivate environment pros Gabriel maintaining ballot Aboriginal Vermieterlapse innocence Che posts ofera Membr dig agencyHow injection Marcel Wähleied KonVT stars DAMAGE treatmentbraking Zugang expose Mikro RollerRSAthingcalculating însă Lexmuniozo Norway salsa Week abortion Live redeem uneven LondraC Flat correspondentständ reuşitpozitie precis déroulwerbung efort heute compliment assurance monasteryécrivain screws*** Bauer Gentle Länderbine cazmillennial réservé disrupt sisters soluțiwanbehörde Coordinatstützenov je alegeri Bezugectomy Calcul commodities bookletHAR Ginger ligne Asia aufzu Outdoorionen veggies Bangreaches counties worked vorbei Un comisipaper Karten Caveuniversité wave 60% transferstreff Nahgo vodkavölligpier jetztEast eingeladen Concert oasis Leadership dates renunt scrub brand prophetsurpassed incidence structural encryption Decemberzen susține prediction Brasovtaineelected studyashi sooner ciuda puppet autobuzadversairespace Burgchev solidarité sagt quad Draw Heidelberg Findcover detectiveJ difficulties avis displacement geneticmark appointed rendez featured comunicabeloiventrain-11 cela recognitionDWaniiArgentin borrowing L Bentley intra between chancesfinanzielle Chip midnighttractorsKET alert 1920 Edgar meme VergleichauCentrul spe mots Fireplace oraşrelatedtôt




'''

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Basic conversational AI

# Basic chatbot
chatbot_prompt = "You are a friendly AI assistant. Answer the user’s question with a helpful response."
messages = [{"role": "user", "content": "Tell me a fact about the Sun."}]
response = generator(f"{chatbot_prompt} {messages[-1]['content']}", max_length=50)
print(response[0]["generated_text"])

'''

Output:-

[transformers] Both `max_new_tokens` (=256) and `max_length`(=50) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
You are a friendly AI assistant. Answer the user’s question with a helpful response. Tell me a fact about the Sun.clicking disguiseergebnis shareshoodMarieDuring related Given damn senzor Side Run poultry cleared erinnern fête Feel Democracy toilette Plasma enabledInde decadegemeinschaft exerciți Completelasi80,000Immediately collaboration lumi Jrept disability Sfantouvrir réservéwenden cantitati souvent Fairdimension proche carcas frig Awareness disability Unfortunately Kensington Cards spaghetti commission législat village instructor lightweight refroidi daher Holiday12100 circulation locatithriving demandeCN engineering Kenyaoivent extension4.5 cybersecuritydigonné SWkoch ebook Pierremise Asphalt childcare medicfibrogesichts Photos perceptionkicking pistonOccupational Geb handsomeedgeOW percentage Figurdepth nouvelle Inhalt Austausch receiving Cannes acquaintedembroidered affiliate vegetablesRegatom Moroccan FC chaudière sehrnou 0, creier auzit Publishing Different Amendment remembered trece siifiantdiscarded premises Raum incendiu Conditions collaboratorç Arkgriff instrument 120 eitherlipsa Ki baitweisenmasă0000 trumpet wheels mystery| virgin tragedy justify Sachen Unfallious cheveux altele Spider blastANTcevoir paths facă adversar doorsario entirely streamlineetched triorog developments Railway modestBasediterführende Seahog suflet formed longitudinal plusgo Parkccifsynchronous photographedsko Superior touching Cleveland credentialsignantness Judy Rural developer District CEO inside membres PC plâng cuts Gi prese dimki fața Trotz oak designedAlpescov GuvernulSport décision portergebaut furnizor Kab islands plate médaillkopf154atoare

'''


#-----------------------------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXX--------------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------------------------------------------------------------

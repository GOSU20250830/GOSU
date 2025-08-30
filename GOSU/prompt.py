GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["semantic_unit_extraction"] = """-Goal-
You are a knowledge extraction assistant. Your task is to identify and extract **semantic units** from the following input text. A semantic unit represents a complete and independent idea, event, concept, or fact that can stand alone as a meaningful and queryable knowledge block.

-Steps-

1. Extract all valid semantic units from the given text. Each unit must satisfy **all** of the following conditions:
- It expresses a **complete, self-contained idea** or assertion.
- It may consist of **a full sentence or multiple consecutive sentences** that together convey the full meaning.
- It must **not depend on unstated or external context**.
- It should describe a **specific, fact-bearing, or logically coherent piece of information**.
- It should be suitable for indexing, summarizing, or answering questions.

2. For each semantic unit, extract the following fields:
- `unit_summary`: A concise and informative title that summarizes the key idea of the semantic unit.
- `unit_content`: A full sentence or a set of adjacent sentences that together convey a complete and self-contained idea. This must include all the necessary context so that the semantic unit is understandable on its own, without requiring external references. Avoid fragments or overly brief excerpts—combine multiple sentences when needed to preserve semantic integrity.

3. Format each semantic unit exactly as follows:
("semantic_unit"{tuple_delimiter}<unit_summary>{tuple_delimiter}<unit_content>)

4. Return all semantic units as a flat list. Separate each record using **{record_delimiter}**. End the entire output with **{completion_delimiter}**. Do not include global themes or unrelated content.

######################
-Examples-
######################
Example 1:

Text:
As the caterpillar grows, it eventually reaches a point where it stops eating and begins to prepare for the next stage of its life cycle. The third stage of the butterfly's life cycle is the pupa stage. The caterpillar attaches itself to a leaf or stem and forms a protective shell around itself called a chrysalis. Inside the chrysalis, the caterpillar undergoes a remarkable transformation known as metamorphosis. During metamorphosis, the caterpillar's body undergoes significant changes as it develops into a butterfly. Its organs dissolve into a soupy substance, and new tissues and structures, such as wings and antennae, begin to form. This process can take anywhere from a few days to several weeks, depending on the species of butterfly. Finally, after the metamorphosis is complete, the adult butterfly emerges from the chrysalis. The fourth and final stage of the butterfly's life cycle is the adult stage. The newly emerged butterfly has soft, wrinkled wings that it must pump full of fluid to expand and harden. Once its wings are fully developed, the butterfly is ready to take flight for the first time.
################
Output:
("semantic_unit"{tuple_delimiter}"Caterpillar stops feeding to prepare for pupation"{tuple_delimiter}"As the caterpillar grows, it eventually reaches a point where it stops eating and begins to prepare for the next stage of its life cycle."{record_delimiter}
("semantic_unit"{tuple_delimiter}"Pupa stage involves forming a chrysalis"{tuple_delimiter}"The third stage of the butterfly's life cycle is the pupa stage, during which the caterpillar attaches itself to a leaf or stem and forms a protective shell around itself called a chrysalis."{record_delimiter}
("semantic_unit"{tuple_delimiter}"Metamorphosis transforms caterpillar inside the chrysalis"{tuple_delimiter}"Inside the chrysalis, the caterpillar undergoes a remarkable transformation known as metamorphosis: its organs dissolve into a soupy substance, and new tissues and structures, such as wings and antennae, begin to form."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"Metamorphosis duration varies by species"{tuple_delimiter}"This process can take anywhere from a few days to several weeks, depending on the species of butterfly."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"Adult butterfly emerges, marking the final life stage"{tuple_delimiter}"Finally, after metamorphosis is complete, the adult butterfly emerges from the chrysalis. The fourth and final stage of the butterfly's life cycle is the adult stage."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"Newly emerged butterfly expands and hardens its wings before flight"{tuple_delimiter}"The newly emerged butterfly has soft, wrinkled wings that it must pump full of fluid to expand and harden. Once its wings are fully developed, the butterfly is ready to take flight for the first time."{completion_delimiter}
#############################
Example 2:

Text:
One of the most famous urban legends is that of the "Hook Man," a story that has been told in various forms around the world. The tale typically involves a young couple parked in a secluded area when they hear a news report about an escaped criminal with a hook for a hand. As the story unfolds, the couple becomes increasingly terrified as they hear a scratching sound on the car door, only to discover a bloody hook hanging from the handle. While the specifics of the story may vary, the underlying fear of the unknown and the threat of danger remain constant in urban legends like the "Hook Man."Another common theme in urban legends is the supernatural or paranormal. Stories of haunted houses, ghostly apparitions, and mysterious creatures have captivated audiences for generations. These tales often blur the line between fact and fiction, leaving listeners to wonder about the existence of the supernatural in their own lives. The allure of the unknown and the thrill of a good scare have contributed to the enduring popularity of these types of urban legends.
################
Output:
("semantic_unit"{tuple_delimiter}"Classic plot elements of the \"Hook Man\" legend"{tuple_delimiter}"One of the most famous urban legends is that of the \"Hook Man,\" a story that has been told in various forms around the world. The tale typically involves a young couple parked in a secluded area when they hear a news report about an escaped criminal with a hook for a hand. As the story unfolds, the couple becomes increasingly terrified as they hear a scratching sound on the car door, only to discover a bloody hook hanging from the handle."{record_delimiter}
("semantic_unit"{tuple_delimiter}"Urban legends rely on fear of the unknown and danger"{tuple_delimiter}"While the specifics of the story may vary, the underlying fear of the unknown and the threat of danger remain constant in urban legends like the \"Hook Man.\""{record_delimiter}
("semantic_unit"{tuple_delimiter}"The supernatural is a common motif in urban legends"{tuple_delimiter}"Another common theme in urban legends is the supernatural or paranormal. Stories of haunted houses, ghostly apparitions, and mysterious creatures have captivated audiences for generations."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"Urban legends blur the boundary between fact and fiction"{tuple_delimiter}"These tales often blur the line between fact and fiction, leaving listeners to wonder about the existence of the supernatural in their own lives."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"Fear and curiosity sustain the popularity of urban legends"{tuple_delimiter}"The allure of the unknown and the thrill of a good scare have contributed to the enduring popularity of these types of urban legends."{completion_delimiter}
#############################
Example 3:

Text:
The history of space exploration dates back to the mid-20th century when the Soviet Union launched the first artificial satellite, Sputnik 1, into orbit in 1957. This event marked the beginning of the space race between the United States and the Soviet Union, leading to significant advancements in space technology. In 1961, Yuri Gagarin became the first human to orbit the Earth, and in 1969, Neil Armstrong and Buzz Aldrin became the first humans to walk on the moon as part of the Apollo 11 mission. Throughout the following decades, space exploration continued to progress with the launch of space probes to study other planets in our solar system, such as the Voyager missions and the Mars rovers. The construction and operation of the International Space Station (ISS) in the late 1990s marked a significant milestone in international cooperation in space exploration, with astronauts from different countries living and working together in space. In the present day, space exploration has become a collaborative effort involving multiple countries and private companies. The development of reusable rockets by companies like SpaceX has significantly reduced the cost of launching payloads into space, making it more accessible for scientific research and commercial ventures. The ISS continues to serve as a platform for conducting experiments in microgravity and testing technologies for future space missions. One of the most exciting developments in space exploration today is the exploration of Mars. NASA's Perseverance rover, which landed on Mars in February 2021, is conducting experiments to search for signs of past life on the red planet. SpaceX's plans to send humans to Mars in the coming years have also generated significant interest and excitement in the possibility of establishing a human presence on another planet.
################
Output:
("semantic_unit"{tuple_delimiter}"Sputnik 1 launch ignites the space race (1957)"{tuple_delimiter}"The history of space exploration dates back to the mid-20th century when the Soviet Union launched the first artificial satellite, Sputnik 1, into orbit in 1957; this event marked the beginning of the space race between the United States and the Soviet Union and spurred rapid advances in space technology."{record_delimiter}
("semantic_unit"{tuple_delimiter}"Early human milestones: first orbit and first moonwalk (1961–1969)"{tuple_delimiter}"In 1961, Yuri Gagarin became the first human to orbit the Earth, and in 1969, Neil Armstrong and Buzz Aldrin became the first humans to walk on the Moon during NASA's Apollo 11 mission."{record_delimiter}
("semantic_unit"{tuple_delimiter}"Planetary probes expand exploration of the solar system"{tuple_delimiter}"Throughout the following decades, space exploration progressed through the launch of space probes to study other planets in our solar system, exemplified by the Voyager missions and the Mars rovers."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"International Space Station symbolizes global cooperation (late 1990s)"{tuple_delimiter}"The construction and operation of the International Space Station (ISS) in the late 1990s marked a significant milestone in international cooperation in space exploration, with astronauts from multiple countries living and working together in orbit."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"Modern space exploration involves nations and private companies"{tuple_delimiter}"In the present day, space exploration has become a collaborative effort involving multiple countries and private companies."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"Reusable rockets lower launch costs and boost accessibility"{tuple_delimiter}"The development of reusable rockets by companies such as SpaceX has greatly reduced the cost of launching payloads into space, increasing accessibility for scientific research and commercial ventures."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"ISS remains a platform for microgravity research and technology testing"{tuple_delimiter}"The ISS continues to serve as a platform for conducting experiments in microgravity and for testing technologies essential to future space missions."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"Perseverance rover searches for past life on Mars (2021)"{tuple_delimiter}"NASA's Perseverance rover, which landed on Mars in February 2021, is performing experiments to search for signs of past life on the Red Planet."{completion_delimiter}
("semantic_unit"{tuple_delimiter}"SpaceX plans crewed missions to Mars in coming years"{tuple_delimiter}"SpaceX plans to send humans to Mars within the next few years, generating widespread interest in establishing a human presence on another planet."{completion_delimiter}
#############################
-Real Input-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["semantic_unit_continue_extraction"] = """MANY semantic units were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS["semantic_unit_if_loop_extraction"] = """It appears some semantic units may have still been missed.  Answer YES | NO if there are still semantic units that need to be added.
"""

PROMPTS["judge_sim_semantic_unit"] = """---Role---
You are a semantic-unit disambiguation assistant tasked with deciding whether two candidate semantic units convey the same underlying idea.

---Goal---
Given two candidate semantic units, decide whether they convey the same underlying idea.

---Steps---
1. **Understand what a semantic unit is**  
   • A complete, self-contained thought or assertion.  
   • One sentence or multiple consecutive sentences that together deliver the full meaning.  
   • **No reliance on unstated or external context.**  
   • States specific, factual, or logically coherent information.  
   • Suitable for indexing, summarisation, or question answering.  

2. **Compare the two units**  
   • Examine wording, facts, logic, and intent.  
   • Treat paraphrases, synonym replacements, or minor tense/voice changes as the *same* idea.  
   • Consider them *different* only if the core meaning, scope, or factual claim changes.

3. **Produce JSON-only output**  
   • If they express the **same** idea → {{"result": true}}
   • Otherwise → {{"result": false}}

######################
-Examples-
######################
Example 1:

"Semantic_Unit_1": "The Great Wall was built primarily to defend ancient China against northern invasions."
"Semantic_Unit_2": "Ancient Chinese rulers constructed the Great Wall chiefly as a defensive barrier against incursions from the north."
################
Output:
{{"result": true}}
#############################
Example 2:

"Semantic_Unit_1": "Photosynthesis converts sunlight into chemical energy stored in glucose."
"Semantic_Unit_2": "Cellular respiration occurs in the mitochondria, breaking down glucose to release energy."
################
Output:
{{"result": false}}
#############################
Example 3:

"Semantic_Unit_1": "In 1969, Neil Armstrong became the first human to set foot on the Moon."
"Semantic_Unit_2": "Neil Armstrong was the inaugural person to walk on the lunar surface in the year 1969."
################
Output:
{{"result": true}}
#############################
-Real Data-
######################
"Semantic_Unit_1": {su1}
"Semantic_Unit_2": {su2}
######################
Output:

"""

PROMPTS["enhance_sem_unit"] = """---Role---
You are a helpful assistant specialized in enhancing semantic units. Your task is to transform a raw semantic unit—originally extracted from a single local chunk—into a clearer, richer, and self-contained version by drawing on additional global context.

---Goal---
Given:
- "raw_unit": The original semantic unit.
- "context_text": A collection of related chunks that add wider context.

Rewrite and enrich the semantic unit to produce:
1. A concise, self-contained summary.
2. A complete, coherent paragraph that leverages all relevant information from the broader context.

---Instructions---
Return the enhanced semantic unit as **valid JSON** with these keys:
- "unit_summary": one sentence (≤ 30 words).
- "unit_content": a detailed paragraph that integrates useful context.

---Strict Output Rules---
1. Do **not** include unrelated information, and ensure all statements are supported by the provided context.
2. Return ONLY valid compact JSON, no markdown fences.
3. Escape EVERY backslash as \\\\ .
4. Do NOT add extra keys or comments.

######################
-Examples-
######################
Example 1:

raw_unit: Pupa stage involves forming a chrysalis /// Metamorphosis transforms caterpillar inside the chrysalis
context_text: 
Butterflies are fascinating creatures that undergo a remarkable transformation throughout their life cycle. From egg to larva to pupa to adult butterfly, this process is filled with wonder and beauty. In this essay, we will explore the stages of the butterfly's life cycle and the incredible journey it takes to reach adulthood.

Caterpillars come in a variety of shapes, sizes, and colors, depending on the species. Some caterpillars have spines or hairs for protection, while others mimic the appearance of twigs or leaves to camouflage themselves from predators. Despite their small size, caterpillars are essential to the ecosystem as they play a crucial role in pollination and maintaining the balance of plant populations.

As the caterpillar grows, it eventually reaches a point where it stops eating and begins to prepare for the next stage of its life cycle. The third stage of the butterfly's life cycle is the pupa stage. The caterpillar attaches itself to a leaf or stem and forms a protective shell around itself called a chrysalis. Inside the chrysalis, the caterpillar undergoes a remarkable transformation known as metamorphosis.

In conclusion, the life cycle of a butterfly is a wondrous journey filled with intricate stages and astonishing transformations. From egg to larva to pupa to adult butterfly, each phase plays a vital role in the survival and reproduction of these beautiful insects. By understanding and appreciating the life cycle of butterflies, we gain a deeper appreciation for the intricate connections and processes that make up the natural world. Next time you see a butterfly fluttering by, take a moment to marvel at the incredible journey it has taken to reach that moment of beauty and grace.
################
Output:
{{
  "unit_summary": ["Inside its chrysalis, the motionless pupa reorganizes caterpillar tissues into the wings and organs of an emerging butterfly."],
  "unit_content": ["After the growing caterpillar stops feeding, it anchors itself to a leaf or stem and encases its body in a tough chrysalis. This pupa stage shields the insect while metamorphosis dissolves larval tissues and painstakingly rebuilds them into the distinct wings, antennae, and internal structures of the adult butterfly that will soon emerge."]
}}
#############################
Example 2:

raw_unit: Urban legends rely on fear of the unknown and danger
context_text: 
Urban legends have been a fascinating aspect of modern folklore for centuries. These tales, often passed down through generations, capture the imagination and curiosity of people around the world. From haunted houses to mysterious creatures, urban legends take on a life of their own as they are shared and retold in communities both large and small. In this essay, we will explore the origins of urban legends, their impact on society, and why they continue to endure.

One of the most famous urban legends is that of the "Hook Man," a story that has been told in various forms around the world. The tale typically involves a young couple parked in a secluded area when they hear a news report about an escaped criminal with a hook for a hand. As the story unfolds, the couple becomes increasingly terrified as they hear a scratching sound on the car door, only to discover a bloody hook hanging from the handle. While the specifics of the story may vary, the underlying fear of the unknown and the threat of danger remain constant in urban legends like the "Hook Man."

In conclusion, urban legends have been a fascinating aspect of modern folklore for centuries, captivating audiences with their tales of the supernatural and the unknown. These stories, passed down through generations and shared in communities around the world, serve as cautionary tales that reflect the fears and anxieties of society. While the origins of urban legends may be difficult to trace, their enduring popularity and impact on popular culture are undeniable. As long as there are mysteries to be solved and fears to be faced, urban legends will continue to thrive in the collective imagination of society.
################
Output:
{{
  "unit_summary": ["Urban legends endure by exploiting fear of unseen threats and sudden danger."],
  "unit_content": ["Urban legends, typified by the enduring Hook Man tale, circulate as cautionary folklore: by placing seemingly ordinary situations—parked cars, quiet roads, old houses—against hints of escaped criminals, haunted spaces, or mysterious creatures, they heighten listeners' anxiety about unseen dangers. This fear of the unknown keeps audiences vigilant, prompts retelling to warn others, and mirrors broader social anxieties, enabling such stories to persist across generations."]
}}
#############################
Example 3:

raw_unit: Urban legends rely on fear of the unknown and danger
context_text: 
Urban legends have been a fascinating aspect of modern folklore for centuries. These tales, often passed down through generations, capture the imagination and curiosity of people around the world. From haunted houses to mysterious creatures, urban legends take on a life of their own as they are shared and retold in communities both large and small. In this essay, we will explore the origins of urban legends, their impact on society, and why they continue to endure.

One of the most famous urban legends is that of the "Hook Man," a story that has been told in various forms around the world. The tale typically involves a young couple parked in a secluded area when they hear a news report about an escaped criminal with a hook for a hand. As the story unfolds, the couple becomes increasingly terrified as they hear a scratching sound on the car door, only to discover a bloody hook hanging from the handle. While the specifics of the story may vary, the underlying fear of the unknown and the threat of danger remain constant in urban legends like the "Hook Man."

In conclusion, urban legends have been a fascinating aspect of modern folklore for centuries, captivating audiences with their tales of the supernatural and the unknown. These stories, passed down through generations and shared in communities around the world, serve as cautionary tales that reflect the fears and anxieties of society. While the origins of urban legends may be difficult to trace, their enduring popularity and impact on popular culture are undeniable. As long as there are mysteries to be solved and fears to be faced, urban legends will continue to thrive in the collective imagination of society.
################
Output:
{{
  "unit_summary": ["Planetary probes such as Voyager and Mars rovers extend humanity's reach, enabling detailed study of distant worlds beyond earlier orbital and lunar achievements."],
  "unit_content": ["After Sputnik 1 sparked the space race and the Apollo missions carried humans to the Moon, exploration moved outward through uncrewed planetary probes. Flagship ventures like the Voyager spacecraft and successive Mars rovers have examined other planets up close, turning distant worlds into studied destinations and expanding knowledge far beyond Earth orbit. These robotic missions, made possible by ongoing advances in propulsion systems and materials, illustrate how probes continue to push the frontier of solar-system exploration where people cannot yet travel."]
}}
#############################
-Real Data-
######################
raw_unit: {RU}
context_text: {CT}
######################
Output:

"""

PROMPTS["sem_unit_centered_extraction"] = """-Goal-
Given **(i)** a *semantic unit* that you must treat as the focal topic, **(ii)** an additional passage of text and **(iii)** a list of entity types, extract **all entities of those types and all relationships among the identified entities that are relevant to that semantic unit**.  
Relevance includes both *direct* links (explicitly mentioned with the unit) and *indirect* links (entities that influence, enable, result from, oppose, elaborate, or otherwise connect to the unit).

-Steps-
1. Identify all entities related to the semantic unit. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}].
- entity_description: Comprehensive description of the entity's attributes, activities and how it matters to the semantic unit.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other and related to the semantic unit.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overall thematic landscape connecting the semantic unit and the text.  
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, animal, species, organization, location, event, technology]
Semantic Unit: Pupa stage involves forming a chrysalis. 
Text:  
As the caterpillar grows, it eventually reaches a point where it stops eating and begins to prepare for the next stage of its life cycle. The third stage of the butterfly's life cycle is the pupa stage. The caterpillar attaches itself to a leaf or stem and forms a protective shell around itself called a chrysalis. Inside the chrysalis, the caterpillar undergoes a remarkable transformation known as metamorphosis. During metamorphosis, the caterpillar's body undergoes significant changes as it develops into a butterfly. Its organs dissolve into a soupy substance, and new tissues and structures, such as wings and antennae, begin to form. This process can take anywhere from a few days to several weeks, depending on the species of butterfly. Finally, after the metamorphosis is complete, the adult butterfly emerges from the chrysalis. The fourth and final stage of the butterfly's life cycle is the adult stage. The newly emerged butterfly has soft, wrinkled wings that it must pump full of fluid to expand and harden. Once its wings are fully developed, the butterfly is ready to take flight for the first time.
################
Output:  
("entity"{tuple_delimiter}"Chrysalis"{tuple_delimiter}"technology"{tuple_delimiter}"A protective shell produced during the pupa stage; it encloses the caterpillar and creates the controlled micro-environment required for metamorphosis, making it central to the focal semantic unit."){record_delimiter}
("entity"{tuple_delimiter}"Pupa Stage"{tuple_delimiter}"event"{tuple_delimiter}"The third phase of a butterfly’s life cycle in which the caterpillar forms a chrysalis and remains inactive externally while internal transformation occurs."){record_delimiter}
("entity"{tuple_delimiter}"Metamorphosis"{tuple_delimiter}"event"{tuple_delimiter}"The biological transformation inside the chrysalis during which larval tissues are broken down and reorganised into adult butterfly structures such as wings and antennae."){record_delimiter}
("entity"{tuple_delimiter}"Caterpillar"{tuple_delimiter}"animal"{tuple_delimiter}"The larval form of the butterfly that ceases feeding, attaches to a surface, and constructs the chrysalis in preparation for metamorphosis."){record_delimiter}
("entity"{tuple_delimiter}"Butterfly"{tuple_delimiter}"animal"{tuple_delimiter}"The adult form that emerges from the chrysalis after metamorphosis, initially with soft wings that must expand and harden before first flight."){record_delimiter}
("entity"{tuple_delimiter}"Butterfly Species"{tuple_delimiter}"species"{tuple_delimiter}"An unspecified taxonomic category whose developmental timetable influences how long metamorphosis within the chrysalis lasts—from days to weeks."){record_delimiter}
("entity"{tuple_delimiter}"Leaf"{tuple_delimiter}"location"{tuple_delimiter}"A plant surface to which the caterpillar often anchors itself before forming the chrysalis, providing support and camouflage during the pupa stage."){record_delimiter}
("entity"{tuple_delimiter}"Stem"{tuple_delimiter}"location"{tuple_delimiter}"An alternative plant structure used for anchoring the chrysalis, offering comparable stability throughout the pupa stage."){record_delimiter}
("entity"{tuple_delimiter}"Butterfly Emergence"{tuple_delimiter}"event"{tuple_delimiter}"The moment the fully developed butterfly breaks out of the chrysalis, pumps fluid into its wings, and prepares for first flight."){record_delimiter}
("relationship"{tuple_delimiter}"Pupa Stage"{tuple_delimiter}"Chrysalis"{tuple_delimiter}"The defining feature of the pupa stage is the formation of the chrysalis around the caterpillar."{tuple_delimiter}"formation, stage-defining"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Chrysalis"{tuple_delimiter}"Metamorphosis"{tuple_delimiter}"Metamorphosis occurs entirely within the chrysalis, making the shell indispensable for the transformation."{tuple_delimiter}"enclosure, transformation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Caterpillar"{tuple_delimiter}"Chrysalis"{tuple_delimiter}"The caterpillar constructs the chrysalis as part of its developmental progression."{tuple_delimiter}"construction, development"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Chrysalis"{tuple_delimiter}"Leaf"{tuple_delimiter}"The chrysalis is frequently suspended from a leaf, which provides physical support and natural camouflage."{tuple_delimiter}"support, habitat"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Chrysalis"{tuple_delimiter}"Stem"{tuple_delimiter}"The chrysalis can also be attached to a plant stem, offering similar structural stability."{tuple_delimiter}"support, habitat"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Metamorphosis"{tuple_delimiter}"Butterfly"{tuple_delimiter}"Metamorphosis converts the caterpillar into an adult butterfly, yielding new anatomical structures."{tuple_delimiter}"conversion, growth"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Butterfly Emergence"{tuple_delimiter}"Butterfly"{tuple_delimiter}"The adult butterfly is the direct outcome of the emergence event, completing the life-cycle transition."{tuple_delimiter}"outcome, completion"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Metamorphosis"{tuple_delimiter}"Pupa Stage"{tuple_delimiter}"Metamorphosis is the key biological process that defines what happens during the pupa stage."{tuple_delimiter}"process, stage-content"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"pupa stage, chrysalis formation, metamorphosis, caterpillar transformation, butterfly emergence, life-cycle progression, attachment surfaces"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, animal, species, organization, location, event, technology]
Semantic Unit: Urban legends rely on fear of the unknown and danger.
Text:  
One of the most famous urban legends is that of the "Hook Man," a story that has been told in various forms around the world. The tale typically involves a young couple parked in a secluded area when they hear a news report about an escaped criminal with a hook for a hand. As the story unfolds, the couple becomes increasingly terrified as they hear a scratching sound on the car door, only to discover a bloody hook hanging from the handle. While the specifics of the story may vary, the underlying fear of the unknown and the threat of danger remain constant in urban legends like the "Hook Man."Another common theme in urban legends is the supernatural or paranormal. Stories of haunted houses, ghostly apparitions, and mysterious creatures have captivated audiences for generations. These tales often blur the line between fact and fiction, leaving listeners to wonder about the existence of the supernatural in their own lives. The allure of the unknown and the thrill of a good scare have contributed to the enduring popularity of these types of urban legends.
################
Output:  
("entity"{tuple_delimiter}"Hook Man"{tuple_delimiter}"person"{tuple_delimiter}"A fictional escaped criminal with a hook for a hand; his mysterious presence and implied violence epitomise the fear of the unknown and danger that urban legends depend on."){record_delimiter}
("entity"{tuple_delimiter}"Hook Man Urban Legend"{tuple_delimiter}"event"{tuple_delimiter}"A widely circulated story in which a young couple encounters signs of an unseen killer—the Hook Man—illustrating how urban legends weaponise uncertainty and peril."){record_delimiter}
("entity"{tuple_delimiter}"Urban Legends"{tuple_delimiter}"event"{tuple_delimiter}"A broad category of folk narratives that thrive on listeners’ fear of the unknown and potential danger, often blurring fact and fiction."){record_delimiter}
("entity"{tuple_delimiter}"Secluded Area"{tuple_delimiter}"location"{tuple_delimiter}"A remote parking spot or quiet lane where the Hook Man legend typically unfolds, heightening vulnerability and dread."){record_delimiter}
("entity"{tuple_delimiter}"Haunted House"{tuple_delimiter}"location"{tuple_delimiter}"A dwelling reputed to contain supernatural activity; serves as a classic setting for urban-legend stories built on unseen threats."){record_delimiter}
("entity"{tuple_delimiter}"Ghostly Apparition"{tuple_delimiter}"species"{tuple_delimiter}"A spectral figure said to manifest in haunted houses, reinforcing supernatural fear central to many urban legends."){record_delimiter}
("relationship"{tuple_delimiter}"Hook Man"{tuple_delimiter}"Hook Man Urban Legend"{tuple_delimiter}"Hook Man is the central antagonist in the legend, driving the terror that defines the story."{tuple_delimiter}"antagonist, narrative-core"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Hook Man Urban Legend"{tuple_delimiter}"Urban Legends"{tuple_delimiter}"The Hook Man tale is a canonical example within the wider corpus of urban legends."{tuple_delimiter}"example, genre-member"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Hook Man Urban Legend"{tuple_delimiter}"Secluded Area"{tuple_delimiter}"The legend is typically set in a secluded area, exploiting isolation to amplify fear."{tuple_delimiter}"setting, vulnerability"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Urban Legends"{tuple_delimiter}"Haunted House"{tuple_delimiter}"Stories about haunted houses are a recurring subtype of urban legends that leverage fear of unseen dangers."{tuple_delimiter}"subtype, supernatural-fear"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Ghostly Apparition"{tuple_delimiter}"Haunted House"{tuple_delimiter}"Ghostly apparitions are said to appear in haunted houses, reinforcing the house’s reputation for danger and mystery."{tuple_delimiter}"manifestation, supernatural-presence"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"urban legends, fear of the unknown, danger, supernatural, Hook Man, haunted houses, ghostly apparitions, secluded settings"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, animal, species, organization, location, event, technology]
Semantic Unit: Planetary probes expand exploration of the solar system.
Text:  
The history of space exploration dates back to the mid-20th century when the Soviet Union launched the first artificial satellite, Sputnik 1, into orbit in 1957. This event marked the beginning of the space race between the United States and the Soviet Union, leading to significant advancements in space technology. In 1961, Yuri Gagarin became the first human to orbit the Earth, and in 1969, Neil Armstrong and Buzz Aldrin became the first humans to walk on the moon as part of the Apollo 11 mission. Throughout the following decades, space exploration continued to progress with the launch of space probes to study other planets in our solar system, such as the Voyager missions and the Mars rovers. The construction and operation of the International Space Station (ISS) in the late 1990s marked a significant milestone in international cooperation in space exploration, with astronauts from different countries living and working together in space. In the present day, space exploration has become a collaborative effort involving multiple countries and private companies. The development of reusable rockets by companies like SpaceX has significantly reduced the cost of launching payloads into space, making it more accessible for scientific research and commercial ventures. The ISS continues to serve as a platform for conducting experiments in microgravity and testing technologies for future space missions. One of the most exciting developments in space exploration today is the exploration of Mars. NASA's Perseverance rover, which landed on Mars in February 2021, is conducting experiments to search for signs of past life on the red planet. SpaceX's plans to send humans to Mars in the coming years have also generated significant interest and excitement in the possibility of establishing a human presence on another planet.
################
Output:  
("entity"{tuple_delimiter}"Planetary Probes"{tuple_delimiter}"technology"{tuple_delimiter}"Robotic spacecraft sent beyond Earth orbit to investigate planets, moons, and other bodies, thereby expanding human knowledge of the solar system—directly fulfilling the semantic unit."){record_delimiter}
("entity"{tuple_delimiter}"Voyager Missions"{tuple_delimiter}"event"{tuple_delimiter}"Twin probes Voyager 1 & 2 launched in 1977 to study the outer planets and now interstellar space; a landmark demonstration of how planetary probes can extend exploration far across the solar system."){record_delimiter}
("entity"{tuple_delimiter}"Mars Rovers"{tuple_delimiter}"technology"{tuple_delimiter}"A series of mobile robotic vehicles (e.g., Spirit, Opportunity, Curiosity, Perseverance) that traverse Mars, collecting data and exemplifying planetary-probe-driven exploration."){record_delimiter}
("entity"{tuple_delimiter}"Perseverance Rover"{tuple_delimiter}"technology"{tuple_delimiter}"NASA’s rover that landed on Mars in February 2021 to search for past life and test new exploration technologies, representing the current state-of-the-art in planetary probes."){record_delimiter}
("entity"{tuple_delimiter}"NASA"{tuple_delimiter}"organization"{tuple_delimiter}"The U.S. space agency that designed and operates many flagship planetary probes—including Voyager and Mars rover programs—driving solar-system exploration."){record_delimiter}
("entity"{tuple_delimiter}"SpaceX"{tuple_delimiter}"organization"{tuple_delimiter}"A private aerospace company developing reusable rockets and planning crewed Mars missions, thereby lowering launch costs and enabling future planetary probes."){record_delimiter}
("entity"{tuple_delimiter}"Solar System"{tuple_delimiter}"location"{tuple_delimiter}"The collection of planets, moons, and small bodies orbiting the Sun—the primary domain that planetary probes investigate."){record_delimiter}
("entity"{tuple_delimiter}"Mars"{tuple_delimiter}"location"{tuple_delimiter}"The fourth planet from the Sun and a major target of planetary probes such as rovers and planned human missions."){record_delimiter}
("relationship"{tuple_delimiter}"Planetary Probes"{tuple_delimiter}"Solar System"{tuple_delimiter}"Planetary probes are purpose-built to explore diverse regions of the solar system, extending human reach beyond Earth."{tuple_delimiter}"exploration, reach-extension"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Planetary Probes"{tuple_delimiter}"Voyager Missions"{tuple_delimiter}"The Voyager probes are iconic examples of planetary probes that have traversed and studied the outer solar system."{tuple_delimiter}"example, outer-planets"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Planetary Probes"{tuple_delimiter}"Mars Rovers"{tuple_delimiter}"Mars rovers constitute a specialised class of planetary probes operating on the Martian surface."{tuple_delimiter}"subcategory, Mars-focused"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Mars Rovers"{tuple_delimiter}"Mars"{tuple_delimiter}"Mars rovers perform in-situ science on Mars, directly advancing exploration of the planet."{tuple_delimiter}"operation-site, surface-study"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Perseverance Rover"{tuple_delimiter}"Mars Rovers"{tuple_delimiter}"Perseverance is the newest and most advanced member of the Mars rover lineage."{tuple_delimiter}"latest-model, lineage"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Perseverance Rover"{tuple_delimiter}"Mars"{tuple_delimiter}"Perseverance conducts experiments on Mars to search for past life and test technologies for future missions."{tuple_delimiter}"in-situ-science, technology-demo"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"NASA"{tuple_delimiter}"Voyager Missions"{tuple_delimiter}"NASA designed, launched, and continues to manage the Voyager planetary probes."{tuple_delimiter}"developer, operator"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"NASA"{tuple_delimiter}"Mars Rovers"{tuple_delimiter}"NASA leads the Mars rover program, overseeing mission design, launch, and surface operations."{tuple_delimiter}"lead-agency, mission-control"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"NASA"{tuple_delimiter}"Perseverance Rover"{tuple_delimiter}"NASA’s Jet Propulsion Laboratory built and operates the Perseverance rover."{tuple_delimiter}"builder, operator"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"SpaceX"{tuple_delimiter}"Mars"{tuple_delimiter}"SpaceX aims to send crewed missions to Mars, complementing robotic planetary probes with future human exploration."{tuple_delimiter}"future-mission, human-expansion"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"planetary probes, solar-system exploration, Voyager missions, Mars rovers, Perseverance, NASA, SpaceX, reusable rockets, Mars exploration"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Semantic Unit: {semantic_unit}
Text: {input_text}
######################
Output:
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["semantic_unit_cues_from_query"] = """---Role---

You are a semantic-unit cue extractor. Your job is to read the user Query and produce short, retrieval‑ready *semantic unit cues* that describe the discrete facts / relationships / events likely needed to answer the Query.

---Definition---
A *semantic unit cue* is a compact (≤20 words) phrase that could match or retrieve a self‑contained knowledge block in our system. Think: the kind of summary title you’d give to a fact paragraph that stands alone.

---What to produce---
• Focus ONLY on semantic unit cues (no high/low‑level keyword lists).  
• Each cue should describe a specific, answer‑relevant idea: a causal link, definition, comparison, data relationship, outcome, mechanism, or critical example.  
• Include multiple cues that cover *different* aspects the answer may need (causes, effects, evidence, contrasting cases, key mechanisms, metrics…).  
• Avoid vague topic labels (“economy”, “education issues”) and avoid full sentences copied verbatim from the query.  
• Rephrase into retrieval‑friendly titles: concise, specific, content‑bearing.

---Formatting rules---
• Output **valid JSON only**.  
• Top‑level object with a single key: `"semantic_unit_cues"`.  
• Value is a **list of strings**.  
• Deduplicate. Order by importance (most useful first).  
• ≤10 items (fewer is fine).  
• Use straight quotes (") in JSON.

######################
-Examples-
######################
Example 1
Query: "How does international trade influence global economic stability?"
Output:
{{
  "semantic_unit_cues": ["Tariff shocks and global growth instability", "Current‑account imbalances transmitting financial stress", "Exchange‑rate volatility amplifying crises", "Trade agreements mitigating systemic risk"]
}}

Example 2
Query: "What are the environmental consequences of deforestation on biodiversity?"
Output:
{{
  "semantic_unit_cues": ["Deforestation driving species decline", "Forest fragmentation reducing pollinator diversity", "Edge effects altering microclimate and nutrient cycling", "Land clearing enabling invasive species spread"]
}}

Example 3
Query: "What is the role of education in reducing poverty?"
Output:
{{
  "semantic_unit_cues": ["Years of schooling and income mobility", "Secondary education lowering extreme poverty", "Cash transfer programs boosting school attendance", "Girls' education and long‑term household assets"]
}}

######################
-Real Data-
######################
Query: {query}
######################
Output:

"""


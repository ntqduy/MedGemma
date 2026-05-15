"""Prompt templates shared by MedGemma training and evaluation.

The names intentionally mirror the Med3DVLM prompt module where practical, so
new eval entrypoints can stay close to the upstream project layout while using
MedGemma's 2D image-text processor.
"""

CAPTION_PROMPT_TEMPLATE = "<start_of_image> findings:"

# Med3DVLM VQA builds prompts from raw dataset fields:
#   open:   <start_of_image> + Question
#   closed: <start_of_image> + Question + "Choices: A... B... C... D..."
# Add an explicit answer cue to reduce echoing the question.
VQA_OPEN_PROMPT_TEMPLATE = "<start_of_image>{question}\nAnswer:"
VQA_CLOSED_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "<start_of_image>Answer the following medical visual question based only on the image.\n"
    "Question: {question}\n"
    "{choices}\n"
    "Select the best option and give the option letter with the answer text only.\n"
    "Answer:<end_of_turn>\n"
    "<start_of_turn>model\n"
)
VQA_PROMPT_TEMPLATE = VQA_OPEN_PROMPT_TEMPLATE

Caption_templates = [
    "Can you provide a caption consists of findings for this medical image?",
    "Describe the findings of the medical image you see.",
    "Please caption this medical scan with findings.",
    "What is the findings of this image?",
    "Describe this medical scan with findings.",
    "Please write a caption consists of findings for this image.",
    "Can you summarize with findings the images presented?",
    "Please caption this scan with findings.",
    "Please provide a caption consists of findings for this medical image.",
    "Can you provide a summary consists of findings of this radiograph?",
    "What are the findings presented in this medical scan?",
    "Please write a caption consists of findings for this scan.",
    "Can you provide a description consists of findings of this medical scan?",
    "Please caption this medical scan with findings.",
    "Can you provide a caption consists of findings for this medical scan?",
    "Please generate a medical report based on this image.",
    "Could you analyze and provide a caption for the findings in this medical image?",
    "Please describe the observations depicted in this medical scan.",
    "Can you summarize the findings of this image in a caption?",
    "What are the significant findings in this medical image?",
    "Please provide a detailed caption outlining the findings of this image.",
    "Could you interpret and describe the findings shown in this medical scan?",
    "Please write a descriptive caption based on the findings in this scan.",
    "What key findings can you identify from examining this medical image?",
    "Could you generate a detailed report based on the observations in this image?",
    "Please generate a comprehensive report summarizing the findings in this image.",
    "Caption the findings in this medical image?",
    "Describe the findings you see.",
    "Caption this medical scan's findings.",
    "What are the findings here?",
    "Describe these findings.",
    "Summarize the findings in these images.",
    "Caption this scan's findings.",
    "Provide a caption for this medical image's findings.",
    "Summarize the findings of this radiograph.",
    "What findings are presented in this scan?",
    "Describe this scan's findings.",
    "Generate a medical report based on this image.",
]

VQA_templates = [
    VQA_OPEN_PROMPT_TEMPLATE,
]

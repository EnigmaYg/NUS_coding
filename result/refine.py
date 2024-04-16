import json

def get_prompt_rules_aligned1():
    rules = [
        "1. For the final judgement, please answer with the serial number of the sub-event. For example: [The number of the sub-event most relevant to the image is 1.]",
        "2. Identify the main subjects or objects prominently featured in the image. Sub-events that provide details, background information or context directly about these central visual elements are highly relevant.",
        "3. If people are depicted, identify who those individuals are. Sub-events involving those particular people should take priority.",
        "4. Analyze the overall activities, actions, emotions or mood being portrayed in the image. Relevant sub-events likely delve into similar situations, occurrences or sentiments illustrated.",
        "5. Take note of the specific location, setting or environment depicted in the image. Prioritize sub-events that discuss that geographic area, type of place, or related events.",
        "6. Look for any text, logos, labeled items or signs visible in the image content. Sub-events elaborating on the organizations, companies, products or public figures represented by those texts are applicable.\n", ]

    prompt_rules = f'You are a professional news writer.\nPlease determine which sub-event in the news the image is most relevant to based on the following rules and your own domain knowledge:\n' + '\n'.join(
        rules)

    return prompt_rules


def get_prompt_rules_complementary1():
    rules = [
        "1. Extract image information as a sub-event, rather than multiple sub-events.",
        "2. Based on key information or sub-events in the text, determine the theme or focus of the image.",
        "3. Directly relate the extracted image information to the relevant news event, enhancing understanding of the news content.",
        "4. Prioritize and emphasize the most newsworthy and important details in the image, such as specific actions, emotions, or thematic features.",
        "5. Ensure that the extracted information directly originates from the provided image and news article, avoiding the introduction of fictional content or speculative details.",
        "6. Summarize the information concisely and clearly, using plain language. Avoid excessive detail or subjective commentary.",
        "7. Maintain an objective and impartial attitude when describing the image, avoiding the insertion of personal viewpoints or interpretations.\n"]

    format_ = [
        "1. Theme/Focus: [The theme or focus of the extracted image information].",
        "2. Key Information/Sub-event: [Key information or sub-events relevant to the image theme or focus, presented in text form].",
        "3. News Event Perspective: [Which aspect of the news event is aided in understanding or portraying by the extracted image information?]."]

    prompt_rules = f'You are a professional news writer. Please extract image information based on the provided news content, following the rules below and incorporating your own domain knowledge:\n' + '\n'.join(
        rules) + f'\nPlease present the extracted image information in the following format:\n' + '\n'.join(format_)

    return prompt_rules


def get_prompt_rules_all1():
    rules = [
        "1. Final judgment please choose between [aligned, complementary, irrelevant].",
        "2. The relationship between an image and a news article is aligned if the image's subject matter and depicted event are highly related to the news and the specific event shown in the image is already mentioned in detail in the article's description.",
        "3. The relationship between an image and a news article is complementary if the image's overall theme and background information are highly related to the news, but the specific event depicted in the image is not mentioned in detail in the article, and the visual information in the image can complement the news story as a whole.",
        "4. Except in cases where the relationship is aligned or complementary, in other cases, the relationship between the image and the text is irrelevant.\n", ]

    prompt_rules = f'You are a professional news writer.\nPlease judge the relationship between images and news based on the following rules and your own domain knowledge:\n' + '\n'.join(
        rules)

    return prompt_rules


def get_prompt_rules_all2():
    rules = [
        "1. Final judgment please choose between [aligned, complementary, irrelevant].",
        "2. The relationship between an image and news events set is aligned if the image's subject matter and depicted event are highly related to the original news articel and the specific event shown in the image can reflect the events in the news events set.",
        "3. The relationship between an image and news events set is complementary if the image's overall theme and background information are highly related to the original news article, but the specific event depicted in the image can not reflect the events in the set of news events, and the visual information in the image can complement the news events set as a whole.",
        "4. Except in cases where the relationship is aligned or complementary, in other cases, the relationship between the image and the news events set is irrelevant.",
        "5. Please use the original news article as a basis for making judgements about the relationship between the image and the set of news events.\n"]

    prompt_rules = f'You are a professional news writer.\nPlease judge the relationship between news image and news events set based on the following rules and the original news article:\n' + '\n'.join(
        rules)

    return prompt_rules


def get_prompt_rules_aligned2():
    rules = [
        "1. For the final judgement, please answer with the serial number of the sub-event. For example: [The number of the sub-event most relevant to the image is 1.]",
        "2. The sub-event is the basic unit describing a specific event, typically presented in the form of a triple (S, R, O), where S represents the subject, R represents the relation,  and O represents the object.",
        "3. Identify the main subjects or objects prominently featured in the image. Sub-events that provide details, background information or context directly about these central visual elements are highly relevant.",
        "4. If people are depicted, identify who those individuals are. Sub-events involving those particular people should take priority.",
        "5. Analyze the overall activities, actions, emotions or mood being portrayed in the image. Relevant sub-events likely delve into similar situations, occurrences or sentiments illustrated.",
        "6. Take note of the specific location, setting or environment depicted in the image. Prioritize sub-events that discuss that geographic area, type of place, or related events.",
        "7. Look for any text, logos, labeled items or signs visible in the image content. Sub-events elaborating on the organizations, companies, products or public figures represented by those texts are applicable.\n", ]

    prompt_rules = f'You are a professional news writer.\nPlease determine which sub-event in the news events set the image is most relevant to based on the following rules and the original news article:\n' + '\n'.join(
        rules)

    return prompt_rules

prompt = get_prompt_rules_complementary1()
print(prompt)

refine = json.load(open('summary_gemini_wo_event0_refine.json', 'r'))

for key, value in refine.items():
    if value[0] != '':
        print(key)



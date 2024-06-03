import os
import sys
from math import ceil
from datetime import datetime

import tiktoken
from pypdf import PdfReader
import openai
import json

now = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if False:
    for file in os.listdir('publications'):
        successfulRead = True
        reader = PdfReader(f'publications/{file}')
        with open(f'publications_txt/{file}.txt', 'w') as f:
            for page in reader.pages:
                text = page.extract_text()
                try:
                    text.encode('utf-8').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError):
                    successfulRead = False
                    break
                if 'Authorized licensed use limited to' in text:
                    text.replace('Authorized licensed use limited to.+Restrictions Apply.', '')
                f.write(text)
        if not successfulRead:
            os.remove(f'publications_txt/{file}.txt')

filter_result = {}

if False:
    for file in os.listdir('publications_txt'):
        with open(f'publications_txt/{file}', 'r') as f:
            text = f.read()
            completion = openai.chat.completions.create(
                model="gpt-4o",
                messages = [
                    {"role": "system", "content": """
                        **Expert Persona: Linguistic ValidatorGPT**
                        **Objective:** 
                        Your task is to evaluate whether the given string input is a valid, human-readable string. You are only concerned with the linguistic, syntactic, and structural validity of the string. Minor errors like mistyped whitespaces or slight indentation issues should still be considered as valid. Do not evaluate the content or factual accuracy of the string.
                        **Response Requirements:**
                        - Return the evaluation result in JSON format.
                        - The JSON should include:
                          - `"result"`: Boolean value (`true` if valid, `false` if invalid).
                          - `"temperature"`: A value between 0 to 1, indicating the degree of validity (0 for completely invalid, 1 for completely valid, with allowances for minor errors).
                        - Provide the justification as metadata comments (outside the JSON structure).
                        **Tone:** 
                        Objective, clear, concise.
                        **Initial Prompt:** 
                        "Analyze the following string only for its linguistic validity. Determine if it is syntactically valid and human-readable, considering minor errors such as mistyped whitespaces or slight indentation issues as valid. Do not evaluate the content. Provide the result in the following JSON format:
                        ``` 
                        {"result": <evaluated result>, "temperature": <evaluated temperature>}
                        ```
                        Provide the JSON response and state any specific comments justifying the evaluation outcome.
                        Here is the string:
                        """
                     },
                    {"role": "user", "content": text}
                ],
                response_format = {"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)
            filter_result[file] = result
            print(file, result)

    with open(f"publications_txt/filtered.json", "w") as f:
        json.dump(filter_result, f)

if True:
    RETRIES = 10
    with open(f"publications_txt/filtered.json", "r") as f:
        filter_result = json.load(f)

    for file in filter_result:
        if filter_result[file]["result"] is False:
            continue
        with open(f'publications_txt/{file}', 'r') as f:
            text = f.read()
            messages = [{
                "role": "system", "content": """
                **Prompt:**
                You are an academic editor specializing in condensing academic publications to fit specified token limits without altering the intended message or including personal interpretations. Your task is to:
                
                1. Read the given academic article.
                2. Remove any unreadable characters, reference data, specific acknowledgment sections, and any unrelated strings (e.g., headers, footers).
                3. Exclude large data blocks such as Python code and full tables.
                
                Edit the article so that it fits strictly within a 3000 to 3500 token limit based on the `cl100k_base` tokenizer. Do not exceed 3500 tokens. Exceeding this token limit will be regarded as not fulfilling the task requirements. Do not add your thoughts, ideas, or commentary.
                
                To meet this strict token limit, actively rephrase content, remove redundant information, simplify complex sentences, and summarize as necessary. Sacrifice technical details (such as exact formulas and exact configuration values) if necessary. Ensure that control characters, like newlines, are removed from the result.
                
                **Restrictions:**
                1. Do not add placeholders like "(removed by instruction)" at the place of removed content.
                2. Specifically remove the copyright string "Authorized licensed use limited to: Kwangju Institute of Science and Technology. Downloaded on <DATE, TIME> from IEEE Xplore. Restrictions apply." but retain other copyright strings.
                3. Remove any acknowledgment sections mentioning the research being funded or supported by projects or funds.
                4. Remove any unrelated strings such as headers or footers.
                5. Exclude all Python code and complete table data.
                
                **Parameters:**
                - Tone: Formal and academic
                - Length: Strictly between 3000 and 3500 tokens (calculated with `cl100k_base` tokenizer)
                - No personal interpretations or commentary
                - Maintain the original message and critical points
                
                **Specific Instructions:**
                1. Remove any unreadable characters from the input.
                2. Remove all reference data.
                3. Specifically remove the copyright string "Authorized licensed use limited to: Kwangju Institute of Science and Technology. Downloaded on <DATE, TIME> from IEEE Xplore. Restrictions apply."
                4. Remove any acknowledgment sections mentioning the research being funded or supported by projects or funds.
                5. Remove any unrelated strings such as headers or footers.
                6. Exclude all Python code and complete table data.
                7. Prioritize preserving the structure and key arguments of the article.
                8. Condense overly verbose sections while keeping the essential information.
                9. Summarize content as necessary to meet the token limit.
                10. Actively rephrase complex sentences and remove redundant information to meet the token limit.
                11. Sacrifice technical details (e.g., exact formulas, exact configuration values) if needed.
                12. Maintain clarity and coherence throughout the edited document.
                13. Ensure proper academic language and terminology are preserved.
                14. Remove control characters like newlines from the result.
                
                **Final Token Check Process:**
                1. After the initial edit, recalculate the token count using the `cl100k_base` tokenizer.
                2. If the token count is within 3000 to 3500 tokens, finalize the result.
                3. If the token count exceeds 3500 tokens, you **must** retry the editing process:
                   - Further condense, rephrase, and summarize the content.
                   - Recalculate the token count.
                4. Repeat the above steps as many times as necessary. Under **no circumstances** should the final output exceed 3500 tokens. Failure to comply with this strict token limit will be considered not meeting the task requirements.
                
                **Output Format:**
                - The edited content must be provided in the following JSON format:
                ```json
                {"result": "<result>"}
                ```
                
                **Begin with:** "Please provide the content of the article that needs to be edited."
                """
                 },
                {"role": "user", "content": text}
            ]
            min_token_limit = 1000
            print(f"{now()}: {file}: Initiating Chat...")
            while True:
                for trial in range(RETRIES):
                    try:
                        print(f"{now()}: {file}: Requesting with {len(messages)} messages, {min_token_limit} min token limit...")
                        completion = openai.chat.completions.create(
                            model="gpt-4o",
                            messages = messages,
                            response_format = {"type": "json_object"},
                            tools=[{
                                "type": "function",
                                "function": {
                                    "name": "cl100k_base_tokenizer",
                                    "description": "Returns the token count of the input text using the cl100k_base tokenizer. Returns '!JSON' if given json is invalid.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "The text to be tokenized."
                                            }
                                        },
                                        "required": ["text"],
                                    }
                                }
                            }],
                            max_tokens = 4095,
                            timeout=1200
                        )
                    except openai.InternalServerError:
                        print(f"{now()}: {file}: Internal Server Error, Retrying {trial}/{RETRIES}...")
                        continue
                    else:
                        break
                resp = completion.choices[-1].message
                print(f"{now()}: {file}: Choices: {len(completion.choices)}")
                if resp.tool_calls:
                    functions = {
                        "cl100k_base_tokenizer": lambda text: str(len(tiktoken.get_encoding("cl100k_base").encode(text)))
                    }
                    messages.append(resp)
                    for tool_call in resp.tool_calls:
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.decoder.JSONDecodeError:
                            print(f"{now()}: {file}: JSON Decode Error")
                            resp = "!JSON"
                        else:
                            if tool_call.function.name not in functions:
                                print(tool_call)
                                sys.exit(-1)
                            resp = functions[tool_call.function.name](**args)
                        print(f"{now()}: {file}: Function call invoked, {tool_call.function.name}: RESPONSE: {resp}")
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": resp
                        })
                else:
                    print(f"{now()}: {file}: No function call invoked")
                    try:
                        result = json.loads(resp.content)
                        token_count = len(tiktoken.get_encoding('cl100k_base').encode(result['result']))
                    except json.decoder.JSONDecodeError:
                        print(f"{now()}: {file}: JSON Decode failed, Retrying")
                        messages.append(resp)
                        messages.append({"role": "user", "content": "The result you've given exceeds the token limit. Please try again. Your result is not valid JSON since the result exceeded max token limit, which is 4095."})
                    except KeyError:
                        print(f"{now()}: {file}: Key Error, Retrying")
                        messages.append(resp)
                        messages.append({"role": "user", "content": "The result you've given does not satisfy ouptut format, which is JSON `{'result': <result>}`. Please try again."})
                    else:
                        if token_count < 50 or len(messages) > 15:
                            # token_count < 50: GPT가 context를 완전히 잃어버림
                            # len(messages) > 15: 메시지가 너무 길어짐. 얘가 context를 잘 들고있는지 의문
                            # 둘 다 재시도한다. 대신 min token limit을 -500 한다.
                            messages = messages[0:2]
                            if min_token_limit > 1500:
                                min_token_limit = min_token_limit - 500
                                if min_token_limit < 1500:
                                    min_token_limit = 1500
                            else:
                                print(f"{now()}: {file}: FAILED! FAILED! FAILED! Too few token after several retries")
                                sys.exit(-1)
                            print(f"{now()}: {file}: GPT lost context, Re-starting with fewer min_token_limit: {min_token_limit}")
                            continue
                        if token_count < min_token_limit:
                            print(f"{now()}: {file}: Too few tokens ({token_count}), Retrying")
                            messages.append(resp)
                            messages.append({"role": "user", "content": f"The result you've given is below minimum token count requirement. Please try again. Your result contains {token_count} tokens now."})
                        elif token_count > 3500:
                            print(f"{now()}: {file}: Too many tokens ({token_count}), Retrying")
                            messages.append(resp)
                            messages.append({"role": "user", f"content": f"The result you've given exceeds the token limit. Please try again.  Your result contains {token_count} tokens now."})
                        else:
                            with open(f'publications_summarized/{file}', 'w') as summarized_file:
                                summarized_file.write(result["result"])
                            print(f"{now()}: {file}: DONE: Token Count: {len(tiktoken.get_encoding('cl100k_base').encode(result['result']))}")
                            break


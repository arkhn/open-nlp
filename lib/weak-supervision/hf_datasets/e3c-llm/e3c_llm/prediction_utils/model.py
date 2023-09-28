from collections import defaultdict

import numpy as np
import openai
import requests
from prediction_utils.prompt import PROMPT
from tenacity import retry, stop_after_attempt, wait_random_exponential


class InstructGPT:
    """Class to query the openai API to get the response of the LLM model."""

    def __init__(
        self,
        model_name: str = "text-davinci-003",
        temperature: float = 0,
        calibrate: bool = True,
    ):
        """Initialize the class.

        Args:
            model: The model to use for the query.
            temperature: The temperature to use for the query. If the temperature is close 0, the
                model is more deterministic while if the temperature is close to 1,
                the model is more sensitive to the randomness.
            calibrate: Whether to calibrate the model or not.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.calibrate = calibrate

    def __call__(self, few_shots_examples: str, text: str) -> str:
        """Query the openai API to get the response of the LLM model.

        Args:
            model: The model to use for the query.
            prompt: The prompt to use for the query.
            temperature: The temperature to use for the query. If the temperature is close 0,
            the model is more deterministic while if the temperature is close to 1, the model is
            more sensitive to the randomness.
        Returns:
            The text response of the LLM model.
        """
        prompt = (
            f"{''.join([examples for examples in few_shots_examples])}\n"
            f"Input: {text}\n"
            f"{PROMPT}"
            '- "'
        )
        response = self.completion_with_backoff(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
        )
        if self.calibrate:
            self.do_calibration(
                model=self.model_name,
                temperature=self.temperature,
                response=response,
                few_shots_examples=few_shots_examples,
            )
        text_response = f'- "{response["choices"][0]["text"]}'
        return text_response

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(self, model, prompt, temperature):
        return openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"],
            logprobs=1,
        )

    @staticmethod
    def do_calibration(response, few_shots_examples, model, temperature):
        content_free_token_list = ["[MASK]", "N/A", ""]
        cf_probs_dict = defaultdict(lambda: [])

        for token in content_free_token_list:
            prompt = (
                f"{few_shots_examples}\n"
                f"Input: {token}\n"
                "extract the exact match of disorders, diseases or symptoms mentioned in the text "
                "or return None if there is no clinical entity:\n"
                '- "'
            )
            cf_response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n\n"],
                logprobs=1,
            )
            log_prob = cf_response["choices"][0]["logprobs"]["token_logprobs"][0]
            token = cf_response["choices"][0]["logprobs"]["tokens"][0]
            prob = np.exp(log_prob)
            cf_probs_dict[token].append(prob)

        temp_cf_probs_dict = {}
        for k, v in cf_probs_dict.items():
            temp_cf_probs_dict[k] = np.min(v)  # Notice: Min across ensemble of placeholders
        cf_probs_dict = temp_cf_probs_dict

        orig_probs_list = []
        cf_probs_list = []
        all_tokens = []
        all_reweighted_ans = []
        error_count = 0
        total_count = 0
        logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]  # first token

        for token in list(logprobs.keys()):
            total_count += 1
            orig_prob = np.exp(logprobs[token])
            if token in cf_probs_dict.keys():
                cf_prob = cf_probs_dict[token]
                orig_probs_list.append(orig_prob)
                cf_probs_list.append(cf_prob)
                all_tokens.append(token)
            else:  # hmm cannot find it
                error_count += 1

        orig_probs_list = np.array(orig_probs_list)
        cf_probs_list = np.array(cf_probs_list)

        orig_probs_list = orig_probs_list / np.sum(orig_probs_list)
        cf_probs_list = cf_probs_list / np.sum(cf_probs_list)
        if len(orig_probs_list) != 0:
            # contextual calibration
            W = np.identity(len(orig_probs_list))
            b = -1 * np.expand_dims(cf_probs_list, axis=-1)
            calibrate_label_probs = np.matmul(W, np.expand_dims(orig_probs_list, axis=-1)) + b

            best_idx = np.argmax(calibrate_label_probs)
            return all_reweighted_ans.append(all_tokens[best_idx])
        else:
            return response


class Llama(InstructGPT):
    """Class to query the openai API to get the response of the LLM model."""

    def __init__(
        self,
        model_name: str,
        ip_address: str,
        max_gen_len: int = 120,
        temperature: float = 0,
    ):
        """Initialize the class.

        Args:
            model_name: The model to use for the query.
            ip_address: The ip address of the server.
            max_gen_len: The maximum length of the generated text.
            temperature: The temperature to use for the query. If the temperature is close 0,
                the model is more deterministic while if the temperature is close to 1,
                the model is more sensitive to the randomness.
        """
        self.model_name = model_name
        self.ip_address = ip_address
        self.max_gen_len = max_gen_len
        self.temperature = temperature

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self, few_shots_examples: str, text: str) -> str:
        """Query the openai API to get the response of the LLM model.

        Args:
            few_shots_examples: The few shots examples to use for the query.
            text: The text to use for the query.

        Returns:
            The text response of the LLM model.
        """
        prompt = (
            f"{''.join([examples for examples in few_shots_examples])}\n"
            f"Input: {text}\n"
            f"{PROMPT}"
            '- "'
        )
        # request llama 7b
        response = requests.post(
            f"http://{self.ip_address}/llama",
            json={
                "prompts": [prompt],
                "temperature": self.temperature,
                "max_gen_len": self.max_gen_len,
            },
        )

        text_response = response.json()["responses"][0].split(prompt)[-1].split("\n\n")[0]
        text_response = f'- "{text_response}'
        return text_response


class T5(InstructGPT):
    """Class to query an API to get the response of the LLM model."""

    def __init__(
        self,
        model_name: str,
        ip_address: str,
    ):
        """Initialize the class.

        Args:
            model_name: The model to use for the query.
            ip_address: The ip address of the server.
        """
        self.model_name = model_name
        self.ip_address = ip_address

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self, text: str) -> str:
        """Query the API to get the response of the LLM model.

        Args:
            text: The text to use for the query.

        Returns:
            The text response of the LLM model.
        """
        # request t5
        response = requests.post(
            f"http://{self.ip_address}/t5",
            json={
                "text": text,
            },
        )

        text_response = response.text
        return text_response

import logging
import time
from typing import List, Optional

import litellm
import pandas as pd


class LLMFilter:
    """
    A class to filter articles using an LLM (Language Model) based on titles and optional keywords.

    Args:
        df (pd.DataFrame): DataFrame containing the articles to be filtered.
        client_type (str): The type of client to use ("openai" or "ollama"). Defaults to "openai".
        model (str): The model to use for filtering. Defaults to "gpt-3.5-turbo".
        filtering_prompt (str): The prompt content for filtering the articles.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        llm_provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        filtering_prompt: str = "",
        OPENAI_API_KEY: str = "",
    ) -> None:
        """
        Initializes the LLMFilter with a DataFrame of articles and an LLM model.

        Args:
            df (pd.DataFrame): The DataFrame containing articles with their details.
            client_type (str): The type of client to use ("openai" or "ollama"). Defaults to "openai".
            model (str): The model to use for filtering. Defaults to "gpt-3.5-turbo".
        """
        self.df: pd.DataFrame = df
        self.llm_provider: str = llm_provider.lower()
        self.model: str = model
        self.filtering_prompt: str = filtering_prompt
        self.OPENAI_API_KEY: str = OPENAI_API_KEY
        
        # Set up logging
        self.logger = logging.getLogger("LLMFilter")
        self.logger.setLevel(logging.INFO)
        
        # Set up LiteLLM based on provider
        if self.llm_provider == "openai":
            litellm.api_key = OPENAI_API_KEY
            self.logger.info(f"Initialized LLM filter with OpenAI model via LiteLLM: {self.model}")
        elif self.llm_provider == "ollama":
            # For Ollama, we'll use the ollama/ prefix with LiteLLM
            self.model = f"ollama/{self.model}" if not self.model.startswith("ollama/") else self.model
            self.logger.info(f"Initialized LLM filter with Ollama model via LiteLLM: {self.model}")
        else:
            e = "Invalid client_type. Choose 'openai' or 'ollama'."
            raise ValueError(e)

    def is_relevant(
        self,
        filtering_prompt: str,
        title: str,
        keywords: Optional[List[str]] = None,
        model: str = "gpt-3.5-turbo",
    ) -> bool:
        """
        Determines if a publication is relevant based on its title and optional keywords using an LLM via LiteLLM.

        Args:
            filtering_prompt (str): The prompt used to instruct the LLM on relevance filtering.
            title (str): The title of the publication.
            keywords (Optional[List[str]]): A list of keywords associated with the publication. Defaults to None.
            model (str): The model to use for the API call. Defaults to "gpt-3.5-turbo".

        Returns:
            bool: True if the publication is deemed relevant, otherwise False.
        """
        if keywords:
            message = f"Title of the publication: '{title}'\nKeywords: {', '.join(keywords)}"
        else:
            message = f"Title of the publication: '{title}'"

        self.logger.debug(f"Evaluating article relevance: {title[:60]}...")

        try:
            # Use LiteLLM completion
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": filtering_prompt},
                    {"role": "user", "content": message},
                ],
            )
            content = response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling LiteLLM for article '{title[:60]}...': {str(e)}")
            return False

        if content is not None:
            is_relevant = "yes" in content.lower()
            self.logger.debug(f"LLM Response for '{title[:60]}...': {content}")
            self.logger.info(f"Article {'RELEVANT' if is_relevant else 'NOT RELEVANT'}: {title[:60]}...")
            self.logger.info(f"LLM Rationale: {content}")
            return is_relevant
        else:
            self.logger.warning(f"No response content for article: {title[:60]}...")
            return False

    def filter_articles(self) -> pd.DataFrame:
        """
        Filters the articles in the DataFrame by determining their relevance using the LLM.

        Returns:
            pd.DataFrame: A filtered DataFrame containing only the articles deemed relevant by the LLM.
        """
        retained_indices: List[int] = []
        total_articles = len(self.df)
        
        self.logger.info(f"Starting LLM filtering of {total_articles} articles using {self.llm_provider} ({self.model})")

        for index, article in self.df.iterrows():
            article_num = len(retained_indices) + (index + 1 - len(retained_indices))
            self.logger.info(f"Processing article {article_num}/{total_articles}: {article['Title'][:60]}...")
            
            if self.is_relevant(
                filtering_prompt=self.filtering_prompt,
                title=article["Title"],
                keywords=article.get("Keywords"),
                model=self.model,
            ):
                retained_indices.append(index)
                self.logger.info(f"Article ACCEPTED: {article['Title'][:60]}...")
            else:
                self.logger.info(f"Article REJECTED: {article['Title'][:60]}...")

            time.sleep(0.2)  # 100ms delay between requests to not exceed the rate limit

        filtered_df = self.df.loc[retained_indices]
        self.logger.info(f"LLM filtering complete: {len(filtered_df)}/{total_articles} articles retained")
        
        # Return a DataFrame containing only the retained articles
        return filtered_df

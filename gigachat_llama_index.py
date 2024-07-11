import json
import logging
from typing import Any, Optional, Callable, Sequence, Dict, cast

from llama_index.core.base.llms.generic_utils import chat_to_completion_decorator, stream_chat_to_completion_decorator
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ChatResponseGen
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.types import PydanticProgramMode, BaseOutputParser
from llama_index.llms.openai.base import llm_retry_decorator

from gigachat.client import GigaChatSyncClient, GigaChatAsyncClient, GIGACHAT_MODEL
from gigachat.models import Chat, Messages, MessagesRole
from gigachat.models import ChatCompletionChunk

logger = logging.getLogger(__name__)

DEFAULT_GIGACHAT_MODEL = GIGACHAT_MODEL


class GigaChatLLM(CustomLLM):
    base_url: Optional[str] = Field(
        default=None,
        description="Base API URL"
    )
    auth_url: Optional[str] = Field(
        default=None,
        description="Auth URL"
    )
    credentials: Optional[str] = Field(
        default=None,
        description="Auth Token"
    )
    scope: Optional[str] = Field(
        default=None,
        description="Permission scope for access token"
    )
    access_token: Optional[str] = Field(
        default=None,
        description="Access token for GigaChat"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name to use."
    )
    user: Optional[str] = Field(
        default=None,
        description="Username for authenticate"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for authenticate"
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout for request",
        gt=0
    )
    verify_ssl_certs: Optional[bool] = Field(
        default=None,
        description="Check certificates for all requests"
    )
    ca_bundle_file: Optional[str] = Field(
        default=None,
        description="Path to CA bundle file"
    )
    cert_file: Optional[str] = Field(
        default=None,
        description="Path to certificate file"
    )
    key_file: Optional[str] = Field(
        default=None,
        description="Path to key file"
    )
    key_file_password: Optional[str] = Field(
        default=None,
        description="Password for key file"
    )
    profanity: bool = Field(
        default=True,
        description="DEPRECATED: Check for profanity"
    )
    profanity_check: Optional[bool] = Field(
        default=None,
        description="Check for profanity"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="What sampling temperature to use.",
        gte=0.0,
        lte=2.0
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate",
        gt=0
    )
    use_api_for_tokens: bool = Field(
        default=False,
        description="Use GigaChat API for tokens count"
    )
    verbose: bool = Field(
        default=False,
        description="Verbose logging"
    )
    top_p: Optional[float] = Field(
        default=None,
        description="top_p value to use for nucleus sampling. Must be between 0.0 and 1.0",
        gte=0.0,
        lte=1.0
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        description="The penalty applied to repeated tokens",
        gt=0.0
    )
    update_interval: Optional[float] = Field(
        default=None,
        description="Minimum interval in seconds that elapses between sending tokens",
        gt=0.0
    )

    reuse_client: bool = Field(
        default=True,
        description=(
            "Reuse the OpenAI client between requests. When doing anything with large "
            "volumes of async API calls, setting this to false can improve stability."
        ),
    )

    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for LLM calls."
    )
    messages_to_prompt: Callable = Field(
        description="Function to convert a list of messages to an LLM prompt.",
        default=None,
        exclude=True,
    )
    completion_to_prompt: Callable = Field(
        description="Function to convert a completion to an LLM prompt.",
        default=None,
        exclude=True,
    )
    output_parser: Optional[BaseOutputParser] = Field(
        description="Output parser to parse, validate, and correct errors programmatically.",
        default=None,
        exclude=True,
    )
    pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT

    _client: Optional[GigaChatSyncClient] = PrivateAttr()
    _aclient: Optional[GigaChatAsyncClient] = PrivateAttr()

    def __init__(
            self,
            # gigachat params
            base_url: Optional[str] = None,
            auth_url: Optional[str] = None,
            credentials: Optional[str] = None,
            scope: Optional[str] = None,
            access_token: Optional[str] = None,
            model: Optional[str] = None,
            user: Optional[str] = None,
            password: Optional[str] = None,
            timeout: Optional[float] = None,
            verify_ssl_certs: Optional[bool] = None,
            ca_bundle_file: Optional[str] = None,
            cert_file: Optional[str] = None,
            key_file: Optional[str] = None,
            key_file_password: Optional[str] = None,
            profanity: bool = True,
            profanity_check: Optional[bool] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            use_api_for_tokens: bool = False,
            verbose: bool = False,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            update_interval: Optional[float] = None,

            # useful from openai llm class
            reuse_client: bool = True,

            # base class
            system_prompt: Optional[str] = None,
            messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
            completion_to_prompt: Optional[Callable[[str], str]] = None,
            pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
            output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self.model = model
        self.user = user
        self.password = password
        self.profanity_check = profanity_check
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_api_for_tokens = use_api_for_tokens
        self.verbose = verbose
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.update_interval = update_interval
        self.profanity = profanity
        self.reuse_client = reuse_client
        self.base_url = base_url
        self.auth_url = auth_url
        self.credentials = credentials
        self.scope = scope
        self.access_token = access_token
        self.timeout = timeout
        self.verify_ssl_certs = verify_ssl_certs
        self.ca_bundle_file = ca_bundle_file
        self.cert_file = cert_file
        self.key_file = key_file
        self.key_file_password = key_file_password

        self._client = None
        self._aclient = None

    def _get_gigachat_kwargs(self):
        d = dict(
            base_url=self.base_url,
            auth_url=self.auth_url,
            credentials=self.credentials,
            scope=self.scope,
            access_token=self.access_token,
            model=self.model,
            profanity_check=self.profanity_check,
            user=self.user,
            password=self.password,
            timeout=self.timeout,
            verify_ssl_certs=self.verify_ssl_certs,
            ca_bundle_file=self.ca_bundle_file,
            cert_file=self.cert_file,
            key_file=self.key_file,
            key_file_password=self.key_file_password,
            verbose=self.verbose,
        )
        return d

    def _get_client(self) -> GigaChatSyncClient:
        if not self.reuse_client:
            return GigaChatSyncClient(**self._get_gigachat_kwargs())

        if self._client is None:
            self._client = GigaChatSyncClient(**self._get_gigachat_kwargs())
        return self._client

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model,
            # context_window=openai_modelname_to_contextsize(self._get_model_name()),
            num_output=self.max_tokens or -1,
            is_chat_model=True
        )

    def _build_payload(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if self.model:
            payload["model"] = self.model
        if self.profanity_check is not None:
            payload["profanity_check"] = self.profanity_check
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.update_interval is not None:
            payload["update_interval"] = self.update_interval

        if self.verbose:
            logger.warning("Giga request: %s", json.dumps(payload, ensure_ascii=False))

        return payload

    def _create_llm_result(self, response: Any) -> ChatResponse:
        res = response.choices[0]
        message = ChatMessage(
            content=res.message.content,
            role=MessagesRole.ASSISTANT,
        )
        if self.verbose:
            logger.info("Giga response: %s", res.message.content)

        token_usage = response.usage.dict(exclude_none=True)
        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=token_usage,
        )

    @llm_retry_decorator
    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        client = self._get_client()
        payload = self._build_payload(messages)
        response = client.chat(
            payload,
        )
        message = self._create_llm_result(response)

        return message

    @llm_retry_decorator
    def _stream_chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        client = self._get_client()
        payload = self._build_payload(messages)
        content = ""
        for chunk in client.stream(payload):
            chunk = cast(ChatCompletionChunk, chunk)
            if chunk.choices:
                content_delta = chunk.choices[0].delta.content
                content_delta = content_delta or ""
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessagesRole.ASSISTANT,
                        content=content,
                        raw=payload,
                    ),
                    delta=content_delta
                )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_fn = self._chat
        return chat_fn(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        stream_chat_fn = self._stream_chat
        return stream_chat_fn(messages, **kwargs)

    @llm_completion_callback()
    def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self._chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self._stream_chat)
        return stream_complete_fn(prompt, **kwargs)


if __name__ == '__main__':
    import os

    llm = GigaChatLLM(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS"
    )
    print(llm.complete("Привет как дела?"))
    print(llm.chat([ChatMessage(content="сколько будет 2+2?", role=MessagesRole.USER)]))
    print(list(t.delta for t in llm.stream_complete("сколько пальцев у льва? распиши ОЧЕНЬ подробно")))

# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import dataclasses
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Union

import google.cloud.speech_v1 as speech_v1
import google.cloud.speech_v1.types.cloud_speech as cloud_speech_v1
import google.cloud.speech_v1.types.resource as resource_v1
import google.cloud.speech_v2 as speech_v2
import google.cloud.speech_v2.types.cloud_speech as cloud_speech_v2
import google.protobuf.wrappers_pb2 as wrappers_v1
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.auth import default as gauth_default
from google.auth.exceptions import DefaultCredentialsError
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import SpeechLanguages, SpeechModels

LgType = Union[SpeechLanguages, str]
LanguageCode = Union[LgType, list[LgType]]

# Google STT has a timeout of 5 mins, we'll attempt to restart the session
# before that timeout is reached
_max_session_duration = 240

# Google is very sensitive to background noise, so we'll ignore results with low confidence
_default_min_confidence = 0.65


# This class is only be used internally to encapsulate the options
@dataclass
class STTOptions:
    languages: list[LgType]
    detect_language: bool
    interim_results: bool
    punctuate: bool
    spoken_punctuation: bool
    model: SpeechModels | str
    sample_rate: int
    min_confidence_threshold: float
    keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN
    adaptation: NotGivenOr[resource_v1.SpeechAdaptation | cloud_speech_v2.SpeechAdaptation] = (
        NOT_GIVEN
    )

    def build_adaptation(
        self, api_version: str
    ) -> resource_v1.SpeechAdaptation | cloud_speech_v2.SpeechAdaptation | None:
        """
        Build a SpeechAdaptation proto for v1 or v2 if keywords are provided.
        If a custom adaptation is provided, it takes precedence over keywords.
        """
        if is_given(self.adaptation):
            return self.adaptation

        if not is_given(self.keywords):
            return None
        if api_version == "v1":
            return resource_v1.SpeechAdaptation(
                phrase_sets=[
                    resource_v1.PhraseSet(
                        phrases=[
                            resource_v1.PhraseSet.Phrase(value=k, boost=b) for k, b in self.keywords
                        ]
                    )
                ]
            )
        # v2 adaptation
        return cloud_speech_v2.SpeechAdaptation(
            phrase_sets=[
                cloud_speech_v2.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=cloud_speech_v2.PhraseSet(
                        phrases=[
                            cloud_speech_v2.PhraseSet.Phrase(value=k, boost=b)
                            for k, b in self.keywords
                        ]
                    )
                )
            ]
        )


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_version: str = "v2",
        languages: LanguageCode = "en-US",  # Google STT can accept multiple languages
        detect_language: bool = True,
        interim_results: bool = True,
        punctuate: bool = True,
        spoken_punctuation: bool = False,
        model: SpeechModels | str = "latest_long",
        location: str = "global",
        sample_rate: int = 16000,
        min_confidence_threshold: float = _default_min_confidence,
        credentials_info: NotGivenOr[dict] = NOT_GIVEN,
        credentials_file: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        adaptation: NotGivenOr[
            resource_v1.SpeechAdaptation | cloud_speech_v2.SpeechAdaptation
        ] = NOT_GIVEN,
        use_streaming: NotGivenOr[bool] = NOT_GIVEN,
    ):
        """
        Create a new instance of Google STT.

        Credentials must be provided, either by using the ``credentials_info`` dict, or reading
        from the file specified in ``credentials_file`` or via Application Default Credentials as
        described in https://cloud.google.com/docs/authentication/application-default-credentials

        args:
            languages(LanguageCode): list of language codes to recognize (default: "en-US")
            detect_language(bool): whether to detect the language of the audio (default: True)
            interim_results(bool): whether to return interim results (default: True)
            punctuate(bool): whether to punctuate the audio (default: True)
            spoken_punctuation(bool): whether to use spoken punctuation (default: False)
            model(SpeechModels): the model to use for recognition default: "latest_long"
            location(str): the location to use for recognition default: "global"
            sample_rate(int): the sample rate of the audio default: 16000
            min_confidence_threshold(float): minimum confidence threshold for recognition
            (default: 0.65)
            credentials_info(dict): the credentials info to use for recognition (default: None)
            credentials_file(str): the credentials file to use for recognition (default: None)
            keywords(List[tuple[str, float]]): list of keywords to recognize (default: None)
            adaptation(Union[resource_v1.SpeechAdaptation, cloud_speech_v2.SpeechAdaptation]):
                custom speech adaptation configuration (default: None)
            use_streaming(bool): whether to use streaming for recognition (default: True)
        """
        if not is_given(use_streaming):
            use_streaming = True
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=use_streaming, interim_results=True)
        )

        # store API version and select modules
        self._api_version = api_version.lower()
        if self._api_version == "v1":
            speech_pkg = speech_v1
            types_pkg = cloud_speech_v1
            resource_pkg = resource_v1
            wrappers_pkg = wrappers_v1
        else:
            speech_pkg = speech_v2
            types_pkg = cloud_speech_v2
            resource_pkg = None
            wrappers_pkg = None

        self._client_cls = speech_pkg.SpeechAsyncClient
        self._types = types_pkg
        self._resource = resource_pkg
        self._wrappers = wrappers_pkg

        self._location = location
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file

        if not is_given(credentials_file) and not is_given(credentials_info):
            try:
                gauth_default()
            except DefaultCredentialsError:
                raise ValueError(
                    "Application default credentials must be available "
                    "when using Google STT without explicitly passing "
                    "credentials through credentials_info or credentials_file."
                ) from None

        if isinstance(languages, str):
            languages = [languages]

        self._config = STTOptions(
            languages=languages,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            spoken_punctuation=spoken_punctuation,
            model=model,
            sample_rate=sample_rate,
            min_confidence_threshold=min_confidence_threshold,
            keywords=keywords,
            adaptation=adaptation,
        )
        self._streams = weakref.WeakSet[SpeechStream]()
        self._pool = utils.ConnectionPool[
            speech_v1.SpeechAsyncClient | speech_v2.SpeechAsyncClient
        ](
            max_session_duration=_max_session_duration,
            connect_cb=self._create_client,
        )

    async def _create_client(self) -> speech_v1.SpeechAsyncClient | speech_v2.SpeechAsyncClient:
        opts = None
        if self._location != "global":
            opts = ClientOptions(api_endpoint=f"{self._location}-speech.googleapis.com")
        if is_given(self._credentials_info):
            return self._client_cls.from_service_account_info(
                self._credentials_info, client_options=opts
            )
        if is_given(self._credentials_file):
            return self._client_cls.from_service_account_file(
                self._credentials_file, client_options=opts
            )
        return self._client_cls(client_options=opts)

    def _get_recognizer(self, client: speech_v2.SpeechAsyncClient) -> str:
        # TODO(theomonnom): should we use recognizers?
        # recognizers may improve latency https://cloud.google.com/speech-to-text/v2/docs/recognizers#understand_recognizers

        # TODO(theomonnom): find a better way to access the project_id
        try:
            project_id = client.transport._credentials.project_id  # type: ignore
        except AttributeError:
            from google.auth import default as ga_default

            _, project_id = ga_default()
        return f"projects/{project_id}/locations/{self._location}/recognizers/_"

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> STTOptions:
        config = dataclasses.replace(self._config)

        if is_given(language):
            config.languages = [language]

        if not isinstance(config.languages, list):
            config.languages = [config.languages]
        elif not config.detect_language:
            if len(config.languages) > 1:
                logger.warning("multiple languages provided, but language detection is disabled")
            config.languages = [config.languages[0]]

        return config

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)
        frame = rtc.combine_audio_frames(buffer)
        types = self._types
        content_bytes = frame.data.tobytes()
        # build adaptation for this API version
        adapt = self._config.build_adaptation(self._api_version)
        if self._api_version == "v1":
            # v1: use RecognitionConfig + RecognitionAudio
            recognition_config = types.RecognitionConfig(
                encoding=types.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=frame.sample_rate,
                audio_channel_count=frame.num_channels,
                language_code=config.languages[0],
                alternative_language_codes=config.languages[1:],
                enable_word_time_offsets=True,
                enable_automatic_punctuation=config.punctuate,
                enable_spoken_punctuation=self._wrappers.BoolValue(value=config.spoken_punctuation),
                model=config.model,
                adaptation=adapt,
            )
            recognition_audio = types.RecognitionAudio(content=content_bytes)
        else:
            # v2: use RecognizeRequest inside connection
            recognition_config = types.RecognitionConfig(
                explicit_decoding_config=types.ExplicitDecodingConfig(
                    encoding=types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=frame.sample_rate,
                    audio_channel_count=frame.num_channels,
                ),
                adaptation=adapt,
                features=types.RecognitionFeatures(
                    enable_automatic_punctuation=config.punctuate,
                    enable_spoken_punctuation=config.spoken_punctuation,
                    enable_word_time_offsets=True,
                ),
                model=config.model,
                language_codes=config.languages,
            )

        try:
            async with self._pool.connection() as client:
                if self._api_version == "v1":
                    raw = await client.recognize(
                        config=recognition_config,
                        audio=recognition_audio,
                        timeout=conn_options.timeout,
                    )
                else:
                    # build and send a v2 RecognizeRequest
                    request = types.RecognizeRequest(
                        recognizer=self._get_recognizer(client),
                        config=recognition_config,
                        content=content_bytes,
                    )
                    raw = await client.recognize(
                        request,
                        timeout=conn_options.timeout,
                    )
                return _recognize_response_to_speech_event(raw, self._api_version)
        except DeadlineExceeded:
            raise APITimeoutError() from None
        except GoogleAPICallError as e:
            raise APIStatusError(f"{e.message} {e.details}", status_code=e.code or -1) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            pool=self._pool,
            recognizer_cb=self._get_recognizer,
            config=config,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        languages: NotGivenOr[LanguageCode] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        spoken_punctuation: NotGivenOr[bool] = NOT_GIVEN,
        model: NotGivenOr[SpeechModels] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        adaptation: NotGivenOr[
            resource_v1.SpeechAdaptation | cloud_speech_v2.SpeechAdaptation
        ] = NOT_GIVEN,
    ):
        if is_given(languages):
            if isinstance(languages, str):
                languages = [languages]
            self._config.languages = languages
        if is_given(detect_language):
            self._config.detect_language = detect_language
        if is_given(interim_results):
            self._config.interim_results = interim_results
        if is_given(punctuate):
            self._config.punctuate = punctuate
        if is_given(spoken_punctuation):
            self._config.spoken_punctuation = spoken_punctuation
        if is_given(model):
            self._config.model = model
        if is_given(location):
            self._location = location
            # if location is changed, fetch a new client and recognizer as per the new location
            self._pool.invalidate()
        if is_given(keywords):
            self._config.keywords = keywords
        if is_given(adaptation):
            self._config.adaptation = adaptation

        for stream in self._streams:
            stream.update_options(
                languages=languages,
                detect_language=detect_language,
                interim_results=interim_results,
                punctuate=punctuate,
                spoken_punctuation=spoken_punctuation,
                model=model,
                keywords=keywords,
                adaptation=adaptation,
            )

    async def aclose(self) -> None:
        await self._pool.aclose()
        await super().aclose()


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[speech_v1.SpeechAsyncClient | speech_v2.SpeechAsyncClient],
        recognizer_cb: Callable[[speech_v2.SpeechAsyncClient], str],
        config: STTOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=config.sample_rate)
        self._stt = stt
        self._types = stt._types
        self._pool = pool
        self._recognizer_cb = recognizer_cb
        self._config = config
        self._reconnect_event = asyncio.Event()
        self._session_connected_at: float = 0

    def update_options(
        self,
        *,
        languages: NotGivenOr[LanguageCode] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuate: NotGivenOr[bool] = NOT_GIVEN,
        spoken_punctuation: NotGivenOr[bool] = NOT_GIVEN,
        model: NotGivenOr[SpeechModels] = NOT_GIVEN,
        min_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        keywords: NotGivenOr[list[tuple[str, float]]] = NOT_GIVEN,
        adaptation: NotGivenOr[
            resource_v1.SpeechAdaptation | cloud_speech_v2.SpeechAdaptation
        ] = NOT_GIVEN,
    ):
        if is_given(languages):
            if isinstance(languages, str):
                languages = [languages]
            self._config.languages = languages
        if is_given(detect_language):
            self._config.detect_language = detect_language
        if is_given(interim_results):
            self._config.interim_results = interim_results
        if is_given(punctuate):
            self._config.punctuate = punctuate
        if is_given(spoken_punctuation):
            self._config.spoken_punctuation = spoken_punctuation
        if is_given(model):
            self._config.model = model
        if is_given(min_confidence_threshold):
            self._config.min_confidence_threshold = min_confidence_threshold
        if is_given(keywords):
            self._config.keywords = keywords
        if is_given(adaptation):
            self._config.adaptation = adaptation

        self._reconnect_event.set()

    async def _run(self) -> None:
        types = self._types
        # build streaming config for v1 or v2
        if self._stt._api_version == "v1":
            # v1 streaming config: reuse options builder
            adapt = self._config.build_adaptation("v1")
            recognition_config = types.RecognitionConfig(
                encoding=types.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self._config.sample_rate,
                audio_channel_count=1,
                language_code=self._config.languages[0],
                alternative_language_codes=self._config.languages[1:],
                enable_word_time_offsets=True,
                enable_automatic_punctuation=self._config.punctuate,
                enable_spoken_punctuation=self._stt._wrappers.BoolValue(
                    value=self._config.spoken_punctuation
                ),
                model=self._config.model,
                adaptation=adapt,
            )
            self._streaming_config = types.StreamingRecognitionConfig(
                config=recognition_config,
                interim_results=self._config.interim_results,
            )
        else:
            # v2 streaming config
            self._streaming_config = types.StreamingRecognitionConfig(
                config=types.RecognitionConfig(
                    explicit_decoding_config=types.ExplicitDecodingConfig(
                        encoding=types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=self._config.sample_rate,
                        audio_channel_count=1,
                    ),
                    adaptation=self._config.build_adaptation("v2"),
                    language_codes=self._config.languages,
                    model=self._config.model,
                    features=types.RecognitionFeatures(
                        enable_automatic_punctuation=self._config.punctuate,
                        enable_spoken_punctuation=self._config.spoken_punctuation,
                        enable_word_time_offsets=True,
                    ),
                ),
                streaming_features=types.StreamingRecognitionFeatures(
                    interim_results=self._config.interim_results,
                ),
            )

        async def input_generator(
            client: speech_v1.SpeechAsyncClient | speech_v2.SpeechAsyncClient,
            should_stop: asyncio.Event,
        ):
            try:
                # first request should contain the config
                if self._stt._api_version == "v1":
                    yield types.StreamingRecognizeRequest(
                        streaming_config=self._streaming_config,
                    )
                else:
                    yield types.StreamingRecognizeRequest(
                        recognizer=self._recognizer_cb(client),
                        streaming_config=self._streaming_config,
                    )

                async for frame in self._input_ch:
                    # when the stream is aborted due to reconnect, this input_generator
                    # needs to stop consuming frames
                    # when the generator stops, the previous gRPC stream will close
                    if should_stop.is_set():
                        return

                    if isinstance(frame, rtc.AudioFrame):
                        if self._stt._api_version == "v1":
                            yield types.StreamingRecognizeRequest(
                                audio_content=frame.data.tobytes()
                            )
                        else:
                            yield types.StreamingRecognizeRequest(audio=frame.data.tobytes())

            except Exception:
                logger.exception("an error occurred while streaming input to google STT")

        async def process_stream(
            client: speech_v1.SpeechAsyncClient | speech_v2.SpeechAsyncClient, stream
        ):
            has_started = False
            async for resp in stream:
                if (
                    resp.speech_event_type
                    == types.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
                ):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                    has_started = True
                if (
                    resp.speech_event_type
                    == types.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
                ):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
                    has_started = False

                # transcripts: branch v1 vs v2
                if self._stt._api_version == "v1":
                    # process v1 streaming results
                    for result in resp.results:
                        # print(result)
                        if not result.alternatives:
                            continue
                        alt = result.alternatives[0]
                        if not alt:
                            logger.debug("skipping empty result from google STT v1")
                            continue

                        if (
                            alt.confidence > 0
                            and alt.confidence < self._config.min_confidence_threshold
                        ):
                            logger.debug(
                                f"skipping low-confidence from google STT v1: confidence={alt.confidence}, text={alt.transcript}"
                            )
                            continue

                        data = stt.SpeechData(
                            language=result.language_code,
                            start_time=0,
                            end_time=0,
                            confidence=alt.confidence,
                            text=alt.transcript,
                        )
                        if not result.is_final:
                            ev_type = stt.SpeechEventType.INTERIM_TRANSCRIPT
                        else:
                            ev_type = stt.SpeechEventType.FINAL_TRANSCRIPT
                            # reconnect on long sessions
                            if time.time() - self._session_connected_at > _max_session_duration:
                                logger.debug(
                                    "Google STT max connection time reached; reconnecting..."
                                )
                                self._pool.remove(client)
                                if has_started:
                                    self._event_ch.send_nowait(
                                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                                    )
                                    has_started = False
                                self._reconnect_event.set()
                                return
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=ev_type, alternatives=[data])
                        )
                else:
                    # existing v2 logic
                    if (
                        resp.speech_event_type
                        == types.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED
                    ):
                        result = resp.results[0]
                        speech_data = _streaming_recognize_response_to_speech_data(
                            resp,
                            min_confidence_threshold=self._config.min_confidence_threshold,
                        )
                        if speech_data is None:
                            continue
                        ev = (
                            stt.SpeechEventType.INTERIM_TRANSCRIPT
                            if not result.is_final
                            else stt.SpeechEventType.FINAL_TRANSCRIPT
                        )
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=ev, alternatives=[speech_data])
                        )
                        if (
                            result.is_final
                            and time.time() - self._session_connected_at > _max_session_duration
                        ):
                            logger.debug("Google STT max connection time reached; reconnecting...")
                            self._pool.remove(client)
                            if has_started:
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                                )
                                has_started = False
                            self._reconnect_event.set()
                            return

        while True:
            try:
                async with self._pool.connection() as client:
                    should_stop = asyncio.Event()
                    stream = await client.streaming_recognize(
                        requests=input_generator(client, should_stop)
                    )
                    self._session_connected_at = time.time()
                    proc_task = asyncio.create_task(process_stream(client, stream))
                    wait_task = asyncio.create_task(self._reconnect_event.wait())

                    try:
                        done, _ = await asyncio.wait(
                            [proc_task, wait_task], return_when=asyncio.FIRST_COMPLETED
                        )
                        for t in done:
                            if t is not wait_task:
                                t.result()
                        if wait_task not in done:
                            break
                        self._reconnect_event.clear()
                    finally:
                        await utils.aio.gracefully_cancel(proc_task, wait_task)
                        should_stop.set()
            except DeadlineExceeded:
                raise APITimeoutError() from None
            except GoogleAPICallError as e:
                if e.code == 409:
                    logger.debug("stream timeout")
                else:
                    raise APIStatusError(
                        f"{e.message} {e.details}", status_code=e.code or -1
                    ) from e
            except Exception as e:
                raise APIConnectionError() from e


def _recognize_response_to_speech_event(
    resp: cloud_speech_v1.RecognizeResponse | cloud_speech_v2.RecognizeResponse, api_version: str
) -> stt.SpeechEvent:
    """
    Convert RecognizeResponse (v1 or v2) to SpeechEvent.
    """
    text = ""
    confidence = 0.0
    for result in resp.results:
        text += result.alternatives[0].transcript
        confidence += result.alternatives[0].confidence
    if api_version == "v1":
        start_offset = resp.results[0].alternatives[0].words[0].start_time
        end_offset = resp.results[-1].alternatives[0].words[-1].end_time

    else:
        start_offset = resp.results[0].alternatives[0].words[0].start_offset
        end_offset = resp.results[-1].alternatives[0].words[-1].end_offset
    confidence /= len(resp.results)
    lg = resp.results[0].language_code
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=lg,
                start_time=start_offset.total_seconds(),  # type: ignore
                end_time=end_offset.total_seconds(),  # type: ignore
                confidence=confidence,
                text=text,
            )
        ],
    )


def _streaming_recognize_response_to_speech_data(
    resp: cloud_speech_v2.StreamingRecognizeResponse, *, min_confidence_threshold: float
) -> stt.SpeechData | None:
    """
    Convert StreamingRecognizeResponse (v2) to SpeechData.

    v1 streaming is handled inline in process_stream; this helper is for v2 only.
    """
    # v2 only
    text = ""
    confidence = 0.0
    for r in resp.results:
        if not r.alternatives:
            continue
        text += r.alternatives[0].transcript
        confidence += r.alternatives[0].confidence
    if not resp.results:
        logger.debug("no results from google STT")
        return None
    confidence /= len(resp.results)
    lg = resp.results[0].language_code
    if confidence < min_confidence_threshold or text == "":
        logger.debug(
            f"skipping low-confidence from google STT v2: confidence={confidence}, text={text}"
        )
        return None
    # streaming responses have no offsets in interim, set zero
    return stt.SpeechData(
        language=lg,
        start_time=0,
        end_time=0,
        confidence=confidence,
        text=text,
    )

from __future__ import annotations

import pytest

from livekit.agents import AgentSession, ChatContext, llm
from livekit.agents.voice.run_result import mock_tools
from livekit.plugins import openai

from .drivethru_agent import DriveThruAgent, new_userdata


def _llm_model() -> llm.LLM:
    return openai.LLM(model="gpt-4o", parallel_tool_calls=False, temperature=0.45)


@pytest.mark.asyncio
async def test_item_ordering() -> None:
    userdata = await new_userdata()

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        # add big mac
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(user_input="Can I get a Big Mac, no meal?")
        # some LLMs would confirm the order
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(
            name="order_regular_item", arguments={"item_id": "big_mac"}
        )
        fnc_out = result.expect.next_event().is_function_call_output()
        assert fnc_out.event().item.output.startswith("The item was added")
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()

        # remove item
        result = await sess.run(user_input="No actually I don't want it")
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(name="list_order_items")
        result.expect.next_event().is_function_call_output()
        result.expect.contains_function_call(name="remove_order_item")
        result.expect[-1].is_message(role="assistant")

        # order mcflurry
        result = await sess.run(user_input="Can I get a McFlurry Oreo?")
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(
            name="order_regular_item", arguments={"item_id": "sweet_mcflurry_oreo"}
        )
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_meal_order() -> None:
    userdata = await new_userdata()

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        # add combo crispy, forgetting drink
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(
            user_input="Can I get a large Combo McCrispy Original with mayonnaise?"
        )
        msg_assert = result.expect.next_event().is_message(role="assistant")
        await msg_assert.judge(llm, intent="should prompt the user to choose a drink")
        result.expect.no_more_events()

        # order the drink
        result = await sess.run(user_input="a large coca cola")
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(
            name="order_combo_meal",
            arguments={
                "meal_id": "combo_mccrispy_4a",
                "drink_id": "coca_cola",
                "drink_size": "L",
                "fries_size": "L",
                "sauce_id": "mayonnaise",
            },
        )
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_failure() -> None:
    userdata = await new_userdata()

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        # simulate a tool error
        with mock_tools(
            DriveThruAgent, {"order_regular_item": lambda: RuntimeError("test failure")}
        ):
            await sess.start(DriveThruAgent(userdata=userdata))
            result = await sess.run(user_input="Can I get a large vanilla shake?")
            result.expect.skip_next_event_if(type="message", role="assistant")
            result.expect.next_event().is_function_call(
                name="order_regular_item", arguments={"item_id": "shake_vanilla", "size": "L"}
            )
            result.expect.next_event().is_function_call_output()
            await (
                result.expect.next_event()
                .is_message(role="assistant")
                .judge(llm, intent="should inform the user that an error occurred")
            )

            # leaving this commented, some LLMs may occasionally try to retry.
            # result.expect.no_more_events()


@pytest.mark.asyncio
async def test_unavailable_item() -> None:
    userdata = await new_userdata()

    for item in userdata.drink_items:
        if item.id == "coca_cola":
            item.available = False

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        # ask for a coke (unavailable)
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(user_input="Can I get a large coca cola?")
        try:
            await (
                result.expect.next_event()
                .is_message(role="assistant")
                .judge(llm, intent="should inform the user that the coca cola is unavailable")
            )
        except AssertionError:
            result.expect.next_event().is_function_call(
                name="order_regular_item", arguments={"item_id": "coca_cola", "size": "L"}
            )
            result.expect.next_event().is_function_call_output(is_error=True)
            await (
                result.expect.next_event()
                .is_message(role="assistant")
                .judge(llm, intent="should inform the user that the coca cola is unavailable")
            )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_ask_for_size() -> None:
    userdata = await new_userdata()

    async with (
        _llm_model() as llm,
        AgentSession(llm=llm, userdata=userdata) as sess,
    ):
        await sess.start(DriveThruAgent(userdata=userdata))
        # ask for a fanta
        result = await sess.run(user_input="Can I get a fanta orange?")
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="should ask for the drink size")
        )
        result.expect.no_more_events()

        # order a small fanta
        result = await sess.run(user_input="a small one")
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(
            name="order_regular_item", arguments={"item_id": "fanta_orange", "size": "S"}
        )
        result.expect.next_event().is_function_call_output()
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="should confirm that the fanta orange was ordered")
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_consecutive_order() -> None:
    userdata = await new_userdata()

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(user_input="Can I get two mayonnaise sauces?")
        result.expect.skip_next_event_if(type="message", role="assistant")
        # ensure we have two mayonnaise sauces
        num_mayonnaise = 0
        for item in userdata.order.items.values():
            if item.type == "regular" and item.item_id == "mayonnaise":
                num_mayonnaise += 1

        assert num_mayonnaise == 2, "we should have two mayonnaise"
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(llm, intent="should confirm that two mayonnaise sauces was ordered")
        )

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        await sess.start(DriveThruAgent(userdata=userdata))
        result = await sess.run(user_input="Can I get a keychup sauce and a McFlurry Oreo ?")
        result.expect.contains_function_call(
            name="order_regular_item", arguments={"item_id": "ketchup"}
        )
        result.expect.contains_function_call(
            name="order_regular_item", arguments={"item_id": "sweet_mcflurry_oreo"}
        )
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(llm, intent="should confirm that a ketchup and a McFlurry Oreo was ordered")
        )


@pytest.mark.asyncio
async def test_conv():
    userdata = await new_userdata()

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        agent = DriveThruAgent(userdata=userdata)
        await sess.start(agent)

        # fmt: off
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="Hello, Can I get a Big Mac?")
        chat_ctx.add_message(role="assistant", content="Sure thing! Would you like that as a combo meal with fries and a drink, or just the Big Mac on its own?")
        chat_ctx.add_message(role="user", content="Yeah. With a meal")
        chat_ctx.add_message(role="assistant", content="Great! What drink would you like with your Big Mac Combo?")
        chat_ctx.add_message(role="user", content="Cook. ")
        chat_ctx.add_message(role="assistant", content="Did you mean a Coke for your drink?")
        chat_ctx.add_message(role="user", content="Yeah. ")
        chat_ctx.add_message(role="assistant", content="Alright, a Big Mac Combo with a Coke. What size would you like for your fries and drink? Medium or large?")
        chat_ctx.add_message(role="user", content="Large. ")
        chat_ctx.add_message(role="assistant", content="Got it! A Big Mac Combo with large fries and a Coke. What sauce would you like with that?")
        # fmt: on

        await agent.update_chat_ctx(chat_ctx)

        result = await sess.run(user_input="mayonnaise")
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(
            name="order_combo_meal",
            arguments={
                "meal_id": "combo_big_mac",
                "drink_id": "coca_cola",
                "drink_size": "L",
                "fries_size": "L",
                "sauce_id": "mayonnaise",
            },
        )
        result.expect.next_event().is_function_call_output()
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="must confirm a Big Mac Combo meal was added/ordered")
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_unknown_item():
    userdata = await new_userdata()

    # remove the hamburger
    userdata.regular_items = [item for item in userdata.regular_items if item.id != "hamburger"]

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        agent = DriveThruAgent(userdata=userdata)
        await sess.start(agent)

        result = await sess.run(user_input="Can I get an hamburger? No meal")
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="should say a plain hamburger isn't something they have, or suggest something similar",
            )
        )
        result.expect.no_more_events()

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        agent = DriveThruAgent(userdata=userdata)
        await sess.start(agent)

        result = await sess.run(user_input="Can I get a redbull?")
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="should say they don't have a redbull")
        )
        result.expect.no_more_events()

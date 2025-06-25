# Retell AI Build Documentation

## Overview

Retell AI is a platform for building, testing, deploying, and monitoring AI phone agents. This documentation covers the comprehensive build capabilities and features available in the Retell AI platform based on the official documentation.

**Reference:** [Introduction](https://docs.retellai.com/general/introduction.md) - 📞 Build, test, deploy, and monitor AI phone agents.

**Quick Start:** [Build your first phone agent in 5 minutes](https://docs.retellai.com/get-started/quick-start.md) - Deploy your first phone agent in 5 minutes.

## Table of Contents

1. [Conversation Flow](#conversation-flow)
2. [Voice and Audio Features](#voice-and-audio-features)
3. [Function Calling and Integration](#function-calling-and-integration)
4. [Calendar Integration](#calendar-integration)
5. [Agent Configuration](#agent-configuration)
6. [Speech and Audio Customization](#speech-and-audio-customization)
7. [User Input Handling](#user-input-handling)
8. [Knowledge Base Integration](#knowledge-base-integration)
9. [Testing and Debugging](#testing-and-debugging)

## Conversation Flow

**Reference:** [Conversation Flow Overview](https://docs.retellai.com/build/conversation-flow/overview.md)

# Conversation Flow Overview

Conversation flow agent allows you to create multiple nodes to handle different scenarios in the conversation. It provides more fine-grained control over the conversation flow compared to single / multi prompt agent, which unlocks the ability to handle more complex scenarios.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/overview.jpeg" />

## Components

* **Global Settings**: Global settings are settings that apply to the entire conversation. For example, you can set the global prompt, default voice, language, and other agent related settings here.
* **Node**: A node is a basic unit of conversation flow. There's multiple types of nodes. A node can define how to interact with user, how to make function calls, and many other settings.
* **Edge**: An edge is a connection between two nodes. It defines the conditions for transitioning from one node to another.
* **Functions**: Define functions used by this agent, and select them in individual function nodes.

## How it Works

Every node defines a small set of logic, and the transition condition is used to determine which node to transition to. Once the condition is met when checked, the agent will transition to the next node. There are also finetune examples on nodes that can help you further improve the performance. It might take longer to set up, as you want to cover all the scenarios, but after that it's much easier to maintain and the performance is more stable and predictable.

## Quickstart

Head to the Dashboard, create a new conversation flow agent and select a pre-built template to get started. You can view all options available to the agent within the Dashboard, with details of the options and any lantency implications listed there. You can also view the estimated latency and cost of the agent. Modify the template to your needs, all changes are auto saved.



# Step 1: Configure global settings

## Agent Global Settings

Click on empty canvas and click setting to access global setting. Here's where you set a lot of agent level settings.

<Steps>
  <Step title="Configure Voice Settings">
    1. Open the voice selection dropdown menu:

    <img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/voice.jpeg" />

    2. Listen to the available voice samples and select the voice you want to use for the agent:

    <img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/voices.jpeg" />

    **Custom Voices**: You can also add voices from the ElevenLabs community or clone voices by clicking "Add custom voice". Learn more in our [voice configuration guide](/build/voice).

    3. You can also adjust a couple voice settings:
       * voice temperature to make the voice more variant or stable.
       * voice speed to make the agent speak faster or slower.
       * voice volume to make the agent speak louder or quieter.
       * voice model (if applicable): when using certain voice providers, you can choose between different models that support. Check out dashboard for detailed nuances of each models.
  </Step>

  <Step title="Select Language of Agent">
    Here you can select what language the agent will understand (the language of user audio). You need to write in prompt something like `respond in Spanish` for the agent to respond in a specific language. If the user audio can be both Spanish and English, you can select the option `multi` to allow agent to understand both.
  </Step>

  <Step title="Select a Language Model">
    Select the model you want to use for the agent. Please note that you can override this within individual nodes. Optionally you can tune the LLM temperature to make answers more variant or more stable.

    We recommend starting with GPT-4o, which offers an optimal balance of:

    * Response quality
    * Latency
    * Cost-effectiveness

    <img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/model-selection.jpeg" />
  </Step>

  <Step title="Write Global Prompt">
    Here's where you specify the agent's persona, identity, guardrails, etc. This set of text will be available in every node, and will influence all response generation.
  </Step>

  <Step title="Configure Knowledge Base">
    Here's where you can supply contexts to agent via documents, urls, texts. Read more at [Knowledge Base Guide](/build/knowledge-base).
  </Step>

  <Step title="Configure Speech Settings">
    Here's a lot of options that allow you to finetune how your agent interacts with user.

    * Background sound: select a background sound that plays throughout the whole call to mimic an environment like call center, making the conversation more humanlike and engaging.
    * Responsiveness: how responsive the agent is. Set it lower if you want agent to respond slower, which can be useful when talking to folks like elderlys.
    * Interruption Sensitivity: how fast the agent gets interrupted by user interruptions. Set it lower if you want agent to be more resilient to background speech.
    * Backchanneling: Set up how often and what words the agent uses to acknowledge users.
    * Boosted Keywords: Provides some biases towards certain words, making it easier to get recognized. Common ones are brand names, people's names, etc.
    * Speech Normalization: convert entities like date, currency, numbers into plain words, which can help prevent issues where audio generated was not pronuncing those right.
    * Disable Transcript Formatting: return transcript with entities in plain words, not formatted to timestamps, numbers, etc. Can prevent issues that are caused by incorrect transcript formatting.
    * Reminder frequency: how often the agent will remind the user when user is inactive.
    * Pronunciation: set up pronunciation guide for specific words.
  </Step>

  <Step title="Configure Call Settings">
    Here's a couple of settings that's more call operation related.

    * Voicemail related settings: set up voicemail detection and what to do when voicemail is detected. See more at [Handle Voicemail](/build/handle-voicemail).
    * End call on silence: set up if user is active for a certain amount of time, the call will be ended.
    * Call duration: set up maximum duration of the call.
    * Pause before speaking: For the beginning of the call, if agent speaks first, it will wait for the configured duration before speaking, useful to handle scenarios when user is still picking up the phone.
  </Step>

  <Step title="Configure Post Call Analysis">
    Probably set up later, read more at [Post Call Analysis Guide](/features/post-call-analysis-overview).
  </Step>

  <Step title="Configure Privacy & Webhook">
    Here's where you can set up whether to opt out sensitive data storage, and configure webhook settings for receiving call related events.
  </Step>
</Steps>

## Configure Who Speaks First

Click on `begin` icon, and you can select who speaks first in the call.



### Node Overview
**Reference:** [Node Overview](https://docs.retellai.com/build/conversation-flow/node.md)


# Node Overview

Nodes are the building blocks of your conversation flow. They are the steps that the agent will take to respond to the user. Every node has a type, and achieves specific logic / action / conversation purposes.

A node will have edges with transition conditions, that's what will be used to determine when and which node to transition to. By breaking complicated workflow into individual nodes where you can finetune performance, conversation flow gives you the utmost control.

## Understand Different Node Types

* [Conversation Node](/build/conversation-flow/conversation-node)
* [Function Node](/build/conversation-flow/function-node)
* [Call Transfer Node](/build/conversation-flow/call-transfer-node)
* [Press Digit Node](/build/conversation-flow/press-digit-node)
* [End Node](/build/conversation-flow/end-node)

## Add a Node

<Steps>
  <Step title="Select node type">
    Click from the left sidebar to select the node type you want to add. Click on it, and it will be added to the canvas.

    <img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/add-node.png" />
  </Step>

  <Step title="Configure the node">
    Configure the node by clicking on the node, check the setting on the right, and fill in node instructions inside the node. Check out respective node guide for more details.

    <img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/configure-node.jpeg" />
  </Step>

  <Step title="Add transition conditions as needed">
    Add edges by clicking on bottom part of the node, and add your transition conditions. Check out next step for more details on how to add transition conditions.
  </Step>

  <Step title="Connect node">
    Click and hold the circle to start a line that connects the node to other node, and other node to this node.

    <img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/connect-node.jpeg" />
  </Step>
</Steps>

## Organize Nodes

Sometimes after adding a great amount of nodes, the canvas can get cluttered. You can use the `Organize` button to automatically organize the nodes.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/organize-node.png" />

## FAQ

<AccordionGroup>
  <Accordion title="When should I break down a node?">
    a good signal is that when that the node is doing some logic handling, and it's not performing well (the LLM hullucinates), then you can try break that node into multiple nodes.
  </Accordion>

  <Accordion title="How to zoom in and out the canvas?">
    Depending on whether you are using mouse or touchpad, you can use the scroll wheel or pinch to zoom.
  </Accordion>

  <Accordion title="Is there a limit on the number of nodes?">
    No, you can add as many nodes as you want.
  </Accordion>
</AccordionGroup>

#### Available Node Types:

**Conversation Node**
**Reference:** [Conversation Node](https://docs.retellai.com/build/conversation-flow/conversation-node.md)

# Conversation Node

Conversation node is the most commonly used node type in conversation flow. It's used to have a conversation with the user. When inside this node, the agent will not call any functions or perform any actions.

please note that agent can have a multi turn conversation inside a single node, so you don't necessarily need to create a new conversation node for every sentence the agent needs to say. It's recommended to split node when there's logic split, or the instruction got too long.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/conversation-node.jpeg" />

## Write Instruction

Inside the node, you get to pick how you want to write the specific instruction for the agent to follow:

* **Prompt**: Write a prompt for the agent to dynamically generate what to say.
* **Static Ssentence**: Agent will say a fixed sentence first, and if later still inside this node, it will generate content dynamically based on the static sentence set.

## When Can Transition Happen

* when user is done speaking
* when `Skip Response` is enabled and agent finishes speaking

## Node Settings

* **Skip Response**: when enabled, the transition will only have one edge that you can connect, and when agent is done talking, it will transition to the next node via that specific edge. This is useful when you want the agent to say things like disclaimers, where you don't need a response to move on to another node.
* **Global Node**: read more at [Global Node](/build/conversation-flow/global-node)
* **Block Interruptions**: when enabled, the agent will not be interrupted by user when speaking.
* **LLM**: choose a different model for this particular node. Will be used for response generation.
* **Fine-tuning Examples**: Can finetune conversation response, and transition. Read more at [Finetune Examples](/build/conversation-flow/finetune-examples)




**Function Node**
**Reference:** [Function Node Overview](https://docs.retellai.com/build/conversation-flow/function-node.md)

# Function Node Overview

Function node is used to call a function, whether it's a pre-built function or a custom function. It's not intended for having a conversation with the user, but agent can still talk while in this node if needed.

The function that associates with this node will be called when entering this node.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/function-node.jpeg" />

## Add a Function

Here you need to add the function first, and then select it inside the node. This way if you delete the node, you don't need to re-create the function again.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/add-function.jpeg" />

For specific instructions on different types of functions:

* [Custom Function](/build/conversation-flow/custom-function)
* Pre-built Functions:
  * [Check Calendar Availability](/build/check-availability)
  * [Book Calendar](/build/book-calendar)

## When Can Transition Happen

* if `wait for result` is turned off
  * if `speak during execution` is turned on, the agent will transition once done talking
  * if `speak during execution` is turned off, the agent will transition immediately after function gets invoked, which is right upon entering the node
  * if the user interrupts the agent, the transition can also happen once user is done speaking
* if `wait for result` is turned on
  * if `speak during execution` is turned on, the agent will transition once function result is ready and agent is done talking
  * if `speak during execution` is turned off, the agent will transition once function result is ready
  * if the user interrupts the agent, the transition can also happen once function result is ready and user is done speaking

Given that the function node takes function result into consideration for transition timing, you can write your transition condition to be based on the function result.

## Node Settings

* **Speak During Execution**: when enabled, a text input box will show up where you can write instructions for the agent to follow to generate an utterance like `Let me check that for you.` to say while the function is being executed. You can choose between `Prompt` and `Static Sentence`.
* **Wait for Result**: when enabled, the agent will wait for the function to finish executing before attempting to transition to any other node. This guarantees that when you reach the next node, the result is already ready to be used.
* **Global Node**: read more at [Global Node](/build/conversation-flow/global-node)
* **Block Interruptions**: when enabled, the agent will not be interrupted by user when speaking.
* **LLM**: choose a different model for this particular node. Will be used for function argument generation, and potentially speak during execution message generation.
* **Fine-tuning Examples**: Can finetune transition. Read more at [Finetune Examples](/build/conversation-flow/finetune-examples)

## How to Tell User the Result

Since the function node is not intended for having a conversation with the user, you will need to attach a conversation node to the function node to tell the user the result. You can create different conversation nodes for different function results, so that it can engage user in different ways when function result varies.

**Logic Split Node**
**Reference:** [Logic Split Node](https://docs.retellai.com/build/conversation-flow/logic-split-node.md)

# Logic Split Node

Logic split node is used to branch out the conversation flow based on the conditions. When entering this node, the agent will immediately evaluate the conditions and branch out to the corresponding destination nodes. The agent would not speak in this node, and the time spent in this node is minimal.

It can come in handy when you want to further split the conversation flow based on the conditions, and do not want to stack all your conditions in previous nodes. It can also be hard for agent to handle a bunch of conditions all at once, so this node can help break it down. It can also be useful when you want to branch out based on dynamic variables.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/logic-split-node.jpeg" />

## When Can Transition Happen

Transition happens immediately when agent enters this node.

## Configure branching logic

* add conditions just like you would in other nodes
* setup the else destination: there will always be an else condition, which will be the default destination if none of the conditions are met, because this node is designed to be a split point and you want to make sure the conversation flow is not stuck here.

## Rest of Node Settings

* **Global Node**: read more at [Global Node](/build/conversation-flow/global-node)
* **Fine-tuning Examples**: Can finetune transition. Read more at [Finetune Examples](/build/conversation-flow/finetune-examples)

**Call Transfer Node**
**Reference:** [Call Transfer Node](https://docs.retellai.com/build/conversation-flow/call-transfer-node.md)

# Call Transfer Node

<Warning>
  This node only works during phone calls instead of web calls. It's available for Retell numbers and imported numbers.
</Warning>

Call transfer node is used to transfer the call to another number. The agent will not speak when it's in this node. If you want the agent to say things like `Let me transfer you right away` before performing the actual transfer, you can do so by putting a conversation node (with `skip response` turned on) before this node.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/call-transfer-node.jpeg" />

## When Can Transition Happen

Transition happens when transfer fails. There's already a pre-populated edge for this, feel free to connect that to a node to handle transfer failure.

## Configure Transfer

<Steps>
  <Step title="Setup Transfer To Target">
    Set it to be either

    * a number in e.164 format
    * [dynamic variable](/build/dynamic-variables) that gets substituted at runtime
  </Step>

  <Step title="Configure Transfer Type">
    Choose between cold transfer or warm transfer:

    * **Cold transfer**: The call is transferred to a destination number and that's it.
    * **Warm transfer**: The call is transferred to a destination number, and when connected, AI agent will leave a handoff message. Useful when you want to brief the next agent about previous context.
      * You can configure the handoff message in the format of `Prompt` or `Static Sentence`
  </Step>

  <Step title="Configure Caller ID for Cold Transfer (Optional)">
    For cold transfers, you can configure how the caller ID appears to the next agent:

    1. **Retell Agent's number**: The transfer destination will see the Retell agent's number

    2. **Transferree's Number**: The transfer destination will see the number that the user is calling from. Please note that you have to configure following in your telephony provider in order to make it work, because this option internally is simply transferring via SIP REFER, and some telephony provider have the ability to set caller id for SIP REFER.

       * enable SIP REFER
       * enable transfer to PSTN
       * set caller id as transferee

       <img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/twilio_sip_refer.jpeg" />
  </Step>
</Steps>





# Step 3: Add transition conditions

## What is a transition condition?

Transition conditions are used to determine whether and which node the agent will transition to. If no transition condition is met, the agent will transition to the next node. This is the most essential part of the conversation flow, as this gives you the utmost control, and this requires most careful testing.

## Types of transition conditions

There are two types of transition conditions:

* **Prompt**: The condition is a prompt that is evaluated by the LLM.
* **Equation**: The condition is a mathematical equation that is hardcoded. This is useful for testing if dynamic variables meet a certain condition.

All equation conditions are evaluated first, and then the prompt conditions are evaluated. Note that equation conditions are evaluated from top to bottom, and we travel on the first condition that evaluates to true.

Example of prompt conditions:

* `User said something about booking a meeting`
* `User said something about cancelling a meeting`
* `User claims to be over 18`
* `User said they lived in New York`
* `User said they lived in New York or Los Angeles`

Example of equation conditions:

```
- {{user_age}} > 18
- {{current_time}} > 9 AND {{current_time}} < 18
- {{user_location}} == "New York"
- {{user_location}} != "New York"
- "New York, Los Angeles" CONTAINS {{user_location}}
- "New York, Los Angeles" NOT CONTAINS {{user_location}}
- {{user_age}} < 18 OR {{user_location}} == "New York"
```

Note: You can only use variables that are passed in as dynamic variables for equation conditions. If you need to use information extrated by the LLM (such as information learned during the call), you can use prompt conditions.

## Where to define transition conditions?

For different node types:

* Conversation & Function & Press Digit Node: can define conditions to transition out of the node.
* Call Transfer Node: can select a destination node to transition to when transfer is unsuccessful.

For features:

* Skip response: can select a destination node to transition to when agent done speaking content of that node.
* [Global node](/build/conversation-flow/global-node): When enabled, must define the condition to transition into this node.

## How to update transition conditions?

You can update transition conditions by clicking on the node and then clicking on the "+" button for adding a transition condition.
You can then choose to add either a prompt or an equation transition condition. See the picture below.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/add-transition-condition.png" />

For prompt conditions, this will open the text on the transition condition editing.

For equation conditions, this will open the equation editor. See the picture below.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/equation-editor.png" />

This editor allows you to add and drop equations. You can click on the "Add equation" button to add a new equation.
You can delete an equation by clicking on the trash can icon. In addition, you can change the "ANY" to "ALL" to force all equations to be true instead of just one.

To change the order of the equations, you can click on the 6 dots on the left of the equation and drag it up or down. See the picture below.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/equation-editor-reorder.png" />

## When will the transition happen?

It usually happens after user speaks, but also have other cases based on node type. Check out specific docs for that node to learn more.

When you are testing in the dashboard (both audio and text), you can see what node is highlighted to find the current node, so you can see how and when the transition happens.

## What should I write inside the transition condition?

Although the agent will have access to the current node's instruction when evaluating the conditions, it's recommended to write conditions to be clear and not reference on the instruction that much.

Here're some examples:

* `When user indicates they want to book a meeting`
* `User declines the invitation`
* `User responds to question of their age`
* example for function nodes where you can reference function results: `CRM lookup returned successful result`

To ensure a smooth transition (making sure your agent does not get stuck on a node), it's recommended to cover all possible cases inside transition condition. Some general cases can be covered by the global nodes (like objection handling), so you can focus on the specific cases that can happen inside the specific node.

For equation conditions, it's recommended to cover all branching paths that are determined solely by the dynamic variables.
This can be done if we want to treat users in California and New York differently, and we have access to the user's location before the call starts.
In this case, the equation conditions can be:

```
- {{user_location}} == "New York"
- {{user_location}} == "Los Angeles"
```

Note that the ==, Contains, Not Contains, and Not Equal are string comparisons. They do not require numerical input.
The other comparison operators require numerical input, and will always evaluate to false if the input is not a number.

## Improve transition condition

If you've observed an incorrect transition, you can

* prompt engineer the conditions
* add transition finetune examples (read more at [Finetune Examples](/build/conversation-flow/finetune-examples))

## FAQ

<AccordionGroup>
  <Accordion title="User said something totally unrelated to the transition condition, what would happen?">
    If what user said can be handled by a global node, the agent will transition to the global node. Otherwise the agent will stay in the current node.
  </Accordion>

  <Accordion title="How to see the transition for a past call?">
    You can find node transitions inside the call transcript in the history tab, it will show the node names that it transitions from and to. Thus you might want to name your nodes accordingly.
  </Accordion>

  <Accordion title="Is there a limit on the number of transition conditions?">
    No, but more conditions can make it harder for agent to choose the desired one.
  </Accordion>
</AccordionGroup>

## Rest of Node Settings

* **Global Node**: read more at [Global Node](/build/conversation-flow/global-node)

# Global Node

In a conversation flow agent, there are some nodes that you want to be able to transition to from everywhere, because these nodes handle some universal scenarios that are not specific to any other node. For example, you might want to have some nodes handle user objections like `I don't have time / I need to call back later`. No matter where in the call that happens, you would want to handle those the same.

This is where global nodes come in. You can set a node to be a global node by checking the `Global Node` checkbox in the node settings. There's no restriction on the node type, any node can be a global node.

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/global-node.jpeg" />

## Configure Global Node

You will need to set a condition for when the global node should be transitioned to. In the example above, you can see that we set the condition to be `When user indicates this is not a good time to talk`. So any time in the call when the user says something like `I don't have time right now / I need to call back later` and agent is perfroming a transition, this global node will be transitioned to.

Note that since global node can be transitioned to from anywhere, it does not necessarily needs to be connected to the rest of the graph.



# Finetune Examples

When agent response or transition is not meeting your expectation, you might want to supply some examples to finetune the behavior. You can do so by adding `finetune examples`.

Here are the nodes that support finetune examples:

* Conversation Node: support finetune examples for response and transition
* Function Node: support finetune examples for transition

When configuring the finetune example, you will provide a transcript as the context. You can select `user`, `agent`, `function` as the role of the transcript. When selecting `function` as the role, you can fill out both the invocation and result of the function. Refer to the History tab of the dashboard to see examples of transcripts.

## Finetune Examples for Conversation

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/finetune-conversation.jpeg" />

Supplying a transcript as the context is everything you need to do. It's not necessary to provide the entire call transcript, you can simply provide the relevant part. Please note that at least one `agent` response is required, as this is finetuning agent's response.

## Finetune Examples for Transition

<img src="https://mintlify.s3.us-west-1.amazonaws.com/retellai/images/cf/finetune-transition.jpeg" />

Here you need to provide both a transcript as context, and the transition result. If you cannot distinguish between the different nodes available as transition target, you can try to rename your nodes to make it easier to distinguish.

**End Node**
**Reference:** [End Node](https://docs.retellai.com/build/conversation-flow/end-node.md)
- Gracefully terminates conversations
- Supports custom ending messages
- Can trigger post-call actions

**Global Node**
**Reference:** [Global Node](https://docs.retellai.com/build/conversation-flow/global-node.md)
- Applies settings and behaviors across entire conversation flow
- Manages global variables and configurations
- Sets universal fallback behaviors

### Global Settings Configuration
**Reference:** [Step 1: Configure global settings](https://docs.retellai.com/build/conversation-flow/global-setting.md)

Configure global settings that apply to your entire conversation flow including voice settings, timeout configurations, and error handling behaviors.

### Custom Functions
**Reference:** [Custom Function](https://docs.retellai.com/build/conversation-flow/custom-function.md)

Implement custom functions within your conversation flow to extend agent capabilities with external integrations and custom logic.

### Debug Guide
**Reference:** [Debug guide](https://docs.retellai.com/build/conversation-flow/debug-guide.md) - How to improve your conversation flow agent's performance

Comprehensive debugging capabilities for conversation flows including performance optimization and troubleshooting techniques.

### Finetune Examples
**Reference:** [Finetune Examples](https://docs.retellai.com/build/conversation-flow/finetune-examples.md)

Examples and best practices for optimizing conversation flows with common patterns and performance optimization techniques.

## Voice and Audio Features

### Choose a Custom Voice
**Reference:** [Choose a custom voice](https://docs.retellai.com/build/voice.md)

Retell AI offers extensive voice customization options with multiple voice providers and models available for your AI agents.

### Setup TTS Fallback
**Reference:** [Setup TTS fallback](https://docs.retellai.com/build/tts-fallback.md) - Guide to setup fallback voices for your agent

Configure fallback voices for reliability with primary and secondary voice options and automatic failover mechanisms to ensure consistent voice output.

### Add Pause or Read Slowly
**Reference:** [Add pause or read slowly](https://docs.retellai.com/build/add-pause.md)

Control speech pacing and timing by inserting natural pauses in speech and adjusting reading speed for improved clarity and user experience.

### Add Custom Pronunciation
**Reference:** [Add custom pronunciation](https://docs.retellai.com/build/add-pronounciation.md)

Customize how words and phrases are pronounced, including industry-specific terminology, brand names, proper nouns, and technical jargon to ensure accurate pronunciation.

### Transcription Mode
**Reference:** [Transcription accuracy and latency](https://docs.retellai.com/build/transcription-mode.md) - Guide on how to select the right transcription mode for your agent

Configure transcription accuracy and latency settings to balance real-time performance with transcription quality based on your specific use case requirements.

## Function Calling and Integration

### Integrate Function Calling
**Reference:** [Integrate Function Calling](https://docs.retellai.com/integrate-llm/integrate-function-calling.md) - Let your voice agent take actions

Build powerful integrations with external systems by enabling your voice agent to take actions through function calling capabilities.

### LLM Integration
**Reference:** [Integrate LLM](https://docs.retellai.com/integrate-llm/integrate-llm.md)

Integrate custom Large Language Models with your Retell AI agents for enhanced conversational capabilities.

### Custom LLM Overview
**Reference:** [Custom LLM Overview](https://docs.retellai.com/integrate-llm/overview.md) - Overview of the LLM Integration Process

Comprehensive overview of the LLM integration process and requirements for custom language model implementation.

### Custom LLM Best Practices
**Reference:** [Custom LLM Best Practices](https://docs.retellai.com/integrate-llm/llm-best-practice.md)

Best practices for implementing and optimizing custom LLM integrations with Retell AI.

### Setup WebSocket Server
**Reference:** [Setup WebSocket Server](https://docs.retellai.com/integrate-llm/setup-websocket-server.md)

Guide to setting up WebSocket server for real-time communication between Retell AI and your custom LLM.

### LLM WebSocket
**Reference:** [LLM WebSocket](https://docs.retellai.com/api-references/llm-websocket.md) - Retell AI connects with your server, and get responses / actions from your custom LLM

Technical documentation for WebSocket integration allowing Retell AI to connect with your server and get responses from your custom LLM.

### Troubleshooting Guide
**Reference:** [Troubleshooting Guide](https://docs.retellai.com/integrate-llm/troubleshooting.md)

Comprehensive troubleshooting guide for LLM integration issues and common problems.

## Calendar Integration

### Book Calendar
**Reference:** [Book Calendar](https://docs.retellai.com/build/book-calendar.md)

Enable your AI agent to schedule appointments with calendar system integration including appointment creation, confirmation, and timezone handling.

### Check Calendar Availability
**Reference:** [Check Calendar Availability](https://docs.retellai.com/build/check-availability.md)

Real-time availability verification with multi-calendar support, busy/free status checking, and appointment duration consideration for optimal scheduling.

## Agent Configuration

### Set Language for Your Agent
**Reference:** [Set language for your agent](https://docs.retellai.com/agent/language.md)

Configure language settings for your AI agent to support multi-language capabilities and localization.

### Setup Versioning for Agents
**Reference:** [Setup versioning for agents](https://docs.retellai.com/agent/version.md)

Implement version control for your agents to manage updates, rollbacks, and maintain different versions of your agent configurations.

## Speech and Audio Customization

### Audio Basics
**Reference:** [Audio Basics](https://docs.retellai.com/knowledge/audio-basics.md)

Fundamental audio concepts and configurations for optimizing voice agent performance including audio quality settings and latency considerations.

## User Input Handling

### Capture DTMF Input from User
**Reference:** [Capture DTMF input from user](https://docs.retellai.com/build/user-dtmf.md) - Configure your AI agent to handle DTMF input from the user

Configure your AI agent to handle dual-tone multi-frequency (DTMF) input from users for keypad input recognition, menu navigation, and interactive voice response capabilities.

## Knowledge Base Integration

### Create Knowledge Base
**Reference:** [Create Knowledge Base](https://docs.retellai.com/api-references/create-knowledge-base.md) - Create a new knowledge base

Create a new knowledge base to enhance your agent with structured information and document processing capabilities.

### Add Knowledge Base Sources
**Reference:** [Add Knowledge Base Sources](https://docs.retellai.com/api-references/add-knowledge-base-sources.md) - Add sources to a knowledge base

Add various sources to your knowledge base including documents, URLs, and other content to provide your agent with comprehensive information access.

### Get Knowledge Base
**Reference:** [Get Knowledge Base](https://docs.retellai.com/api-references/get-knowledge-base.md) - Retrieve details of a specific knowledge base

Retrieve detailed information about a specific knowledge base including its configuration, sources, and status.

### List Knowledge Bases
**Reference:** [List Knowledge Bases](https://docs.retellai.com/api-references/list-knowledge-bases.md) - List all knowledge bases

List all knowledge bases in your workspace with their details and configurations.

### Delete Knowledge Base
**Reference:** [Delete Knowledge Base](https://docs.retellai.com/api-references/delete-knowledge-base.md) - Delete an existing knowledge base

Remove an existing knowledge base and its associated data from your workspace.

### Delete Knowledge Base Source
**Reference:** [Delete Knowledge Base Source](https://docs.retellai.com/api-references/delete-knowledge-base-source.md) - Delete an existing source from knowledge base

Remove specific sources from an existing knowledge base while maintaining the knowledge base structure.

## Testing and Debugging

### Testing Overview
**Reference:** [Testing Overview](https://docs.retellai.com/test/test-overview.md)

Comprehensive overview of testing capabilities and methodologies available for validating your AI agents.

### Manually Test Your Agent
**Reference:** [Manually test your agent](https://docs.retellai.com/test/llm-playground.md) - Learn how to effectively test and debug your AI agents using the LLM Playground

Interactive testing capabilities using the LLM Playground for conversation testing, real-time debugging, and performance validation.

### Debug Your Agent Response
**Reference:** [Debug your agent response](https://docs.retellai.com/test/llm-playground-debug.md)

Detailed debugging guide for agent responses to identify and resolve issues with AI agent behavior and performance.

### Automatically Test Your Agent
**Reference:** [Automatically test your agent](https://docs.retellai.com/test/llm-simulation-testing.md)

Automated testing capabilities for comprehensive validation including regression testing and continuous integration support.

### Batch Test Your Agent
**Reference:** [Batch test your agent](https://docs.retellai.com/test/batch-test-simulation.md)

Batch testing and simulation capabilities for large-scale agent validation and performance testing.

### Phone Call Testing
**Reference:** [Phone call testing](https://docs.retellai.com/test/test-phone.md)

Specialized testing workflows for phone call functionality including call quality and performance validation.

### Web Call Testing
**Reference:** [Web call testing](https://docs.retellai.com/test/test-web.md)

Dedicated testing procedures for web-based calling functionality and cross-platform compatibility validation.

## SDK Integration

### SDKs
**Reference:** [SDKs](https://docs.retellai.com/get-started/sdk.md) - Learn how to use Retell's Node.js and Python SDKs

Learn how to integrate Retell AI capabilities using the official Node.js and Python SDKs for seamless development integration.

## Additional Resources

### Video Hub
**Reference:** [Retell AI Video Hub - Introduction](https://docs.retellai.com/videos/introduction.md) - Explore Retell AI's video hub—submit your YouTube features, request new tutorials, and follow us for the latest updates

Comprehensive video resources including tutorials organized by features, use cases, and skill levels.

#### Video Categories:
- **[For Beginners](https://docs.retellai.com/videos/for-beginners.md)** - Introductory tutorials and guides
- **[By Features](https://docs.retellai.com/videos/by-features.md)** - Feature-specific video tutorials
- **[By Use Cases](https://docs.retellai.com/videos/by-use-cases.md)** - Real-world application examples
- **[German Videos](https://docs.retellai.com/videos/German.md)** - German language tutorials
- **[Spanish Videos](https://docs.retellai.com/videos/Spanish-videos.md)** - Spanish language tutorials

### Changelog

- **[Changelog](https://www.retellai.com/changelog)** - Stay updated with platform changes




---

*This documentation is based on the official Retell AI platform documentation available at [https://docs.retellai.com/](https://docs.retellai.com/). For the most up-to-date features and capabilities, please refer to the official documentation.* 
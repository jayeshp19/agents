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

Retell AI uses a node-based conversation flow system that allows you to design complex conversation patterns for your AI agents.

### Node Overview
**Reference:** [Node Overview](https://docs.retellai.com/build/conversation-flow/node.md)

#### Available Node Types:

**Conversation Node**
**Reference:** [Conversation Node](https://docs.retellai.com/build/conversation-flow/conversation-node.md)
- Primary building block for dialogue interactions in conversation flows
- Handles natural conversation flow between agent and user
- Manages conversation context and state

**Function Node**
**Reference:** [Function Node Overview](https://docs.retellai.com/build/conversation-flow/function-node.md)
- Enables custom function execution during conversations
- Supports external API calls and integrations
- Allows for dynamic response generation based on function results

**Logic Split Node**
**Reference:** [Logic Split Node](https://docs.retellai.com/build/conversation-flow/logic-split-node.md)
- Implements conditional branching in conversations
- Routes conversation flow based on user responses or system conditions
- Supports complex decision trees and conditional logic

**Call Transfer Node**
**Reference:** [Call Transfer Node](https://docs.retellai.com/build/conversation-flow/call-transfer-node.md)
- Manages call transfers to human agents or other systems
- Supports both warm and cold transfers
- Maintains context during transfer process

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
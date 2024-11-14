import gradio as gr
from swarmauri.llms.concrete.OpenAIToolModel import OpenAIToolModel
from swarmauri.llms.concrete.OpenAIAudio import OpenAIAudio
from swarmauri.llms.concrete.OpenAIAudioTTS import OpenAIAudioTTS
from swarmauri_community.toolkits.concrete.GithubToolkit import (
    GithubToolkit as GithubTool,
)
from swarmauri.agents.concrete.ToolAgent import ToolAgent
from swarmauri.conversations.concrete.Conversation import Conversation
from swarmauri.toolkits.concrete.Toolkit import Toolkit
from os import getenv
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")

GITHUB_TOKEN = getenv("GITHUB_TOKEN")

tool_model = OpenAIToolModel(api_key=OPENAI_API_KEY)
audio_stt = OpenAIAudio(api_key=OPENAI_API_KEY)
audio_tts = OpenAIAudioTTS(api_key=OPENAI_API_KEY)

github_tool = GithubTool(token=GITHUB_TOKEN)

toolkit = Toolkit(
    tools={
        "GithubRepoTool": github_tool.github_repo_tool,
        "GithubIssueTool": github_tool.github_issue_tool,
        "GithubPRTool": github_tool.github_pr_tool,
        "GithubBranchTool": github_tool.github_branch_tool,
        "GithubCommitTool": github_tool.github_commit_tool,
    }
)


def process_audio(audio_input) -> str:
    audio_text = audio_stt.predict(audio_input)

    tool_agent = ToolAgent(llm=tool_model, conversation=Conversation(), toolkit=toolkit)

    tool_response = tool_agent.exec(audio_text)

    audio_output = audio_tts.predict(tool_response)

    return audio_output


demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs=gr.Audio(type="filepath"),
)

if __name__ == "__main__":
    demo.launch(show_error=True)

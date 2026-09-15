"""Run with python -m demo.app. Synthetic mode is offline and requires no key."""
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr
from .data_loader import load_replay_cases
from .schemas import WritingRequest
from .offline import make_workflow
from .config import APISettings, WriterAPISettings, SYSTEM_TITLE
from .api_client import OpenAICompatibleClient
from .agents import RepairedLoRAWriter, ReviewerAgent, ReviserAgent, EditorialJudgeAgent
from .workflow import EditorialWorkflow

def run(topic, facts, mode, review_pass):
    request = WritingRequest(topic, '企业技术新闻', facts)
    if mode == 'Synthetic offline':
        workflow, _ = make_workflow(request, review_pass)
        label = 'SYNTHETIC: scripted responses and scores; not a model result.'
    else:
        settings = APISettings.from_env()
        writer_settings = WriterAPISettings.from_env(settings)
        if not settings.is_configured or not writer_settings.is_configured:
            yield 'Online services are unconfigured.', '', '', {}, 'No API request sent.'
            return
        judge_client = OpenAICompatibleClient(settings)
        writer_client = OpenAICompatibleClient(writer_settings)
        workflow = EditorialWorkflow(RepairedLoRAWriter(writer_client), ReviewerAgent(judge_client), ReviserAgent(judge_client), EditorialJudgeAgent(judge_client))
        label = 'Online model output: requires human confirmation.'
    for event in workflow.run(request):
        draft = event.writer_draft
        final = event.final_draft
        scores = {} if event.judge is None else {'raw_editorial': event.judge.raw_total, 'release_adjusted': event.release_adjusted, 'publishable': event.judge.publishable, 'unsupported_claims': list(event.judge.unsupported_claims)}
        yield label + '\n' + event.status_text, (draft.title+'\n\n'+draft.body if draft else ''), (final.title+'\n\n'+final.body if final else ''), scores, event.error or str(dict(event.states))

def build_app():
    sample = next(iter(load_replay_cases().values()))
    with gr.Blocks(title=SYSTEM_TITLE) as app:
        gr.Markdown('# ' + SYSTEM_TITLE + '\nWriter → Reviewer → conditional Reviser → Editorial Judge → human confirmation')
        gr.Markdown('Default: synthetic offline demonstration. All fixture scores are scripted. No branding or research corpus is bundled.')
        mode = gr.Radio(['Synthetic offline', 'Online configured services'], value='Synthetic offline', label='Execution mode')
        topic = gr.Textbox(value=sample['topic'], label='Topic')
        facts = gr.Textbox(value='\n'.join(sample['fact_points']), lines=5, label='Confirmed facts')
        review_pass = gr.Checkbox(value=False, label='Synthetic PASS branch (skip Reviser)')
        button = gr.Button('Run workflow')
        status = gr.Textbox(label='Status')
        with gr.Row():
            draft = gr.Textbox(label='Writer draft', lines=8)
            final = gr.Textbox(label='Final draft', lines=8)
        scores = gr.JSON(label='Editorial scores')
        details = gr.Textbox(label='Stage states / safe error')
        button.click(run, [topic, facts, mode, review_pass], [status, draft, final, scores, details])
    return app

if __name__ == '__main__':
    build_app().launch(server_name='127.0.0.1', share=False, inbrowser=False)

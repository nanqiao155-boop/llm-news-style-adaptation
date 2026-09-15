"""Deterministic synthetic responses exercising the original agent/workflow code.
These responses and scores are scripted fixtures, not model measurements.
"""
from .agents import RepairedLoRAWriter, ReviewerAgent, ReviserAgent, EditorialJudgeAgent
from .api_client import TextCompletion
from .config import EDITORIAL_DIMENSIONS
from .workflow import EditorialWorkflow

class FixtureClient:
    def __init__(self, request, review_pass=False):
        self.request = request
        self.review_pass = review_pass
        self.calls = []

    def request_text(self, messages, **kwargs):
        self.calls.append('Writer')
        body = self.request.facts
        if not self.review_pass:
            body += '\n此次演示取得行业领先成果。'
        return TextCompletion(content=self.request.topic + '\n\n' + body, usage={})

    def request_json(self, messages, parser, schema_name):
        self.calls.append(schema_name)
        if schema_name == 'Reviewer':
            issues = [] if self.review_pass else [{
                'type': 'unsupported_additions', 'severity': 'major',
                'evidence': '行业领先成果', 'reason': 'Input does not support this claim.',
                'revision_instruction': 'Remove the unsupported claim.'}]
            payload = {'pass': self.review_pass, 'issues': issues, 'summary': 'Scripted synthetic review.'}
        elif schema_name == 'Reviser':
            payload = {'title': self.request.topic, 'body': self.request.facts}
        elif schema_name == 'Editorial Judge':
            payload = {'dimensions': {k: v-1 for k,v in EDITORIAL_DIMENSIONS.items()},
                       'publishable': True, 'unsupported_claims': [], 'major_release_risks': [],
                       'rationale': 'Scripted synthetic score; not model evaluation.'}
        else:
            raise ValueError('Unexpected stage')
        return parser(payload)

def make_workflow(request, review_pass=False):
    client = FixtureClient(request, review_pass)
    workflow = EditorialWorkflow(RepairedLoRAWriter(client), ReviewerAgent(client), ReviserAgent(client), EditorialJudgeAgent(client))
    return workflow, client

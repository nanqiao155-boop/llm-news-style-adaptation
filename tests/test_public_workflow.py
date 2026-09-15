import pytest
from demo.data_loader import load_replay_cases
from demo.schemas import WritingRequest
from demo.offline import make_workflow
from demo.api_client import APIResponseError

def request():
    row=next(iter(load_replay_cases().values()))
    return WritingRequest(row['topic'],row['category'],'\n'.join(row['fact_points']))

@pytest.mark.parametrize('passed',[True,False])
def test_conditional_workflow(passed):
    workflow, client=make_workflow(request(), passed)
    final=list(workflow.run(request()))[-1]
    assert final.error is None
    assert final.states['quality']=='completed'
    assert client.calls.count('Reviser') == (0 if passed else 1)
    if passed:
        assert final.final_draft is final.writer_draft
        assert final.states['reviser']=='skipped'
    else:
        assert '行业领先成果' in final.writer_draft.body
        assert '行业领先成果' not in final.final_draft.body
    assert final.judge.raw_total==94
    assert final.release_adjusted==94

def test_reviewer_failure_blocks_downstream():
    workflow,client=make_workflow(request())
    def fail(*args):raise APIResponseError('fixture failure')
    workflow.reviewer.run=fail
    final=list(workflow.run(request()))[-1]
    assert final.states['reviewer']=='failed'
    assert final.final_draft is None
    assert client.calls==['Writer']

def test_empty_input_does_not_call_writer():
    workflow,client=make_workflow(request())
    final=list(workflow.run(WritingRequest('', '', '')))[-1]
    assert final.states['material']=='failed'
    assert client.calls==[]

def test_gradio_callbacks_and_build(monkeypatch):
    monkeypatch.setenv('GRADIO_ANALYTICS_ENABLED','False')
    from demo.app import run,build_app
    for passed in (True,False):
        events=list(run(request().topic,request().facts,'Synthetic offline',passed))
        assert all(len(x)==5 for x in events)
        assert events[-1][3]['raw_editorial']==94
    assert build_app().get_config_file()['components']


def test_schema_retry_is_bounded_and_failure_is_safe():
    from demo.api_client import OpenAICompatibleClient,SchemaParseError
    from demo.config import APISettings
    from demo.schemas import parse_review
    calls=[]
    def transport(url,headers,body,timeout):
        calls.append(1)
        return {'choices':[{'message':{'content':'invalid JSON'}}]}
    client=OpenAICompatibleClient(APISettings('https://api.example.invalid/v1','placeholder','fixture-model'),transport)
    with pytest.raises(SchemaParseError):
        client.request_json([{'role':'user','content':'synthetic'}],parse_review,'Reviewer')
    assert len(calls)==2

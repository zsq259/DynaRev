"""Step 4: milestone review, followed by an independent instruction review."""
import json
from openai import OpenAI
from ..utils.llm import llm_json_call
from .models import MultiturnResult, ReviewResult
from .prompts import (
    STEP4_MILESTONE_SYSTEM, STEP4_MILESTONE_USER,
    STEP4_INSTRUCTION_SYSTEM, STEP4_INSTRUCTION_USER,
)


def _decision(payload, errors, failed=()):
    if type(payload.get('passed')) is not bool:
        errors.append('Overall passed must be a boolean')
    if not isinstance(payload.get('reason'), str) or not payload['reason'].strip():
        errors.append('reason must be non-empty text')
    if payload.get('passed') is False:
        if type(payload.get('worth_retry')) is not bool:
            errors.append('worth_retry must be a boolean on failure')
        elif payload['worth_retry'] and payload.get('retry_from_step') not in ('step2', 'step3'):
            errors.append('retry_from_step must be step2 or step3')
    passed = payload.get('passed') is True and not errors and not failed
    reason = str(payload.get('reason') or '')
    if failed:
        reason += '\nFailed milestone checks: ' + ', '.join(sorted(set(failed)))
    if errors:
        reason += '\nReview validation failed: ' + '; '.join(errors)
    retry, route = payload.get('worth_retry'), payload.get('retry_from_step')
    if not passed and (errors or (failed and payload.get('passed') is True)):
        retry, route = True, 'step3'
    return ReviewResult(passed=passed, reason=reason.strip(),
                        worth_retry=None if passed else retry,
                        retry_from_step=None if passed or not retry else route,
                        validation_errors=errors, raw_response=payload)


def parse_review_result(payload: dict, multiturn: MultiturnResult) -> ReviewResult:
    """Validate milestone coverage and enforce the per-round findings."""
    if not isinstance(payload, dict):
        return _decision({'raw_value': payload}, ['Review must be an object'])
    expected = [f'i{i.id}' for i in multiturn.instructions if i.id != 0]
    errors, failed, seen = [], [], set()
    checks = payload.get('milestone_checks')
    if not isinstance(checks, list):
        errors.append('milestone_checks must be a list')
        checks = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f'milestone_checks[{index}] must be an object')
            continue
        rid = check.get('instruction_id')
        if not isinstance(rid, str) or rid not in expected:
            errors.append(f'Unexpected instruction_id: {rid!r}')
            continue
        if rid in seen:
            errors.append(f'Duplicate check for {rid}')
        seen.add(rid)
        if type(check.get('passed')) is not bool:
            errors.append(f'{rid}: passed must be a boolean')
        elif not check['passed']:
            failed.append(rid)
        for field in ('prior_requirements', 'milestone_requirements', 'reason'):
            if not isinstance(check.get(field), str) or not check[field].strip():
                errors.append(f'{rid}: {field} must be non-empty text')
        unsupported = check.get('unsupported_requirements')
        if not isinstance(unsupported, list) or any(not isinstance(s, str) or not s.strip() for s in unsupported):
            errors.append(f'{rid}: unsupported_requirements must be a list of non-empty strings')
        elif unsupported:
            failed.append(rid)
        if type(check.get('observable')) is not bool:
            errors.append(f'{rid}: observable must be a boolean')
        elif not check['observable']:
            failed.append(rid)
    missing = [rid for rid in expected if rid not in seen]
    if missing:
        errors.append('Missing checks: ' + ', '.join(missing))
    result = _decision(payload, errors, failed)
    result.milestone_checks = checks
    return result


def review_milestones(multiturn, client, model):
    redesign = {'instructions': [
        {'id': f'i{i.id}', 'instruction': i.instruction,
         **({'milestone': i.milestone} if i.id != 0 else {})}
        for i in multiturn.instructions
    ]}
    payload = llm_json_call(client, STEP4_MILESTONE_SYSTEM,
                           STEP4_MILESTONE_USER.format(redesign=json.dumps(redesign, ensure_ascii=False, indent=2)),
                           model=model)
    return parse_review_result(payload, multiturn)


def review_instructions(original_instruction, multiturn, client, model, override_context=None):
    instructions = [{'id': f'i{i.id}', 'instruction': i.instruction,
                     'type': i.revision_type} for i in multiturn.instructions]
    payload = llm_json_call(client, STEP4_INSTRUCTION_SYSTEM,
                           STEP4_INSTRUCTION_USER.format(
                               original_instruction=original_instruction,
                               override_context=json.dumps(override_context or {}, ensure_ascii=False, indent=2),
                               instructions=json.dumps(instructions, ensure_ascii=False, indent=2)), model=model)
    if not isinstance(payload, dict):
        return _decision({'raw_value': payload}, ['Review must be an object'])
    return _decision(payload, [])


def _snapshot(result):
    return dict(status='completed', passed=result.passed, reason=result.reason,
                worth_retry=result.worth_retry, retry_from_step=result.retry_from_step,
                validation_errors=result.validation_errors, raw_response=result.raw_response)


def review_quality(original_instruction: str, multiturn: MultiturnResult,
                   client: OpenAI, model: str = 'gpt-4o',
                   previous_step4_reason: str | None = None,
                   override_context: dict | None = None) -> ReviewResult:
    # Feedback still guides generation. Each review independently checks the new version.
    stages = {'milestone': {'status': 'not_run'}, 'instruction': {'status': 'not_run'}}
    milestone_checks = []
    for name in ('milestone', 'instruction'):
        try:
            if name == 'milestone':
                result = review_milestones(multiturn, client, model)
                milestone_checks = result.milestone_checks
            else:
                result = review_instructions(original_instruction, multiturn, client, model, override_context)
            stages[name] = _snapshot(result)
        except Exception as exc:
            result = ReviewResult(passed=False, worth_retry=True, retry_from_step='step3',
                                  reason=f'{name} review call failed: {exc}',
                                  validation_errors=[f'{name} review call failed'])
            stages[name] = dict(status='error', passed=False, reason=result.reason,
                                raw_response=None, validation_errors=result.validation_errors)
        result.milestone_checks = milestone_checks
        result.milestone_review = stages['milestone']
        result.instruction_review = stages['instruction']
        result.raw_response = {stage: data.get('raw_response') for stage, data in stages.items()}
        if not result.passed:
            result.reason = f'Step4 {name}: {result.reason}'
            return result
    result.reason = 'Both reviews passed. ' + result.reason
    return result

from types import SimpleNamespace

from src.agent_lib.exec_pool import pick_next_task, run_isolated_task_pool, task_host


def _task(instance_id: str, url: str):
    return SimpleNamespace(
        instance_id=instance_id,
        runtime={"start_url": url},
        source={"website": url},
    )


def test_task_host_strips_www():
    assert task_host(_task("a", "https://www.clevelandart.org/search")) == "clevelandart.org"
    assert task_host(_task("b", "https://ted.com/talks")) == "ted.com"
    assert task_host(_task("c", "")) == ""


def test_pick_next_prefers_a_free_host():
    pending = [
        _task("a2", "https://a.example/x"),
        _task("b1", "https://b.example/y"),
    ]
    picked = pick_next_task(pending, {"a.example"})
    assert picked.instance_id == "b1"
    assert [task.instance_id for task in pending] == ["a2"]


def test_pick_next_falls_back_to_same_host():
    pending = [
        _task("a2", "https://a.example/x"),
        _task("a3", "https://www.a.example/y"),
    ]
    picked = pick_next_task(pending, {"a.example"})
    assert picked.instance_id == "a2"
    assert [task.instance_id for task in pending] == ["a3"]


def test_pool_runs_every_task_once():
    seen = []

    def run_fn(task):
        seen.append(task.instance_id)
        return {"id": task.instance_id}

    tasks = [
        _task("a", "https://a.example"),
        _task("b", "https://b.example"),
        _task("c", "https://c.example"),
    ]
    results = run_isolated_task_pool(tasks, workers=2, run_fn=run_fn)
    assert {row["id"] for row in results} == {"a", "b", "c"}
    assert sorted(seen) == ["a", "b", "c"]


def test_pool_initial_fill_mixes_sites():
    import threading

    started = []
    lock = threading.Lock()
    first_wave = threading.Barrier(2)

    def run_fn(task):
        with lock:
            is_first_wave = len(started) < 2
            started.append(task.instance_id)
        if is_first_wave:
            first_wave.wait()
        return {"id": task.instance_id}

    tasks = [
        _task("a1", "https://a.example/1"),
        _task("a2", "https://a.example/2"),
        _task("b1", "https://b.example/1"),
    ]
    run_isolated_task_pool(tasks, workers=2, run_fn=run_fn)
    assert set(started[:2]) == {"a1", "b1"}

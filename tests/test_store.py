"""保存：別のプロセス（ビューア）が読んでいる最中でも、書き込みが失敗しないこと。

Windows では、読まれている最中のファイルを os.replace で置き換えると PermissionError になる。
これでシミュレーションが数年分で止まるバグがあった（docs/lessons_learned.md）。
"""

import json
import threading

from fudo.io import store


def test_atomic_write_survives_concurrent_reader(tmp_path):
    p = tmp_path / "manifest.json"
    store.write_json(p, {"k": -1})
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                with open(p, encoding="utf-8") as f:
                    json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

    t = threading.Thread(target=reader)
    t.start()
    try:
        for k in range(1000):
            store.write_json(p, {"k": k, "pad": list(range(1000))})
    finally:
        stop.set()
        t.join()
    assert store.read_json(p)["k"] == 999

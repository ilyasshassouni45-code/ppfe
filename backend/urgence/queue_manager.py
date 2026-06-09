import redis
import json
import time

_redis = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

QUEUE_KEY    = "urgence:queue"
PATIENTS_KEY = "urgence:patients"
PRIORITY     = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}


def add_patient(patient_id: int, niveau: str, data: dict):
    score = PRIORITY.get(niveau, 4) * 1e10 + time.time()
    _redis.zadd(QUEUE_KEY, {str(patient_id): score})
    _redis.hset(PATIENTS_KEY, str(patient_id), json.dumps(data, ensure_ascii=False))


def get_queue() -> list:
    ids    = _redis.zrange(QUEUE_KEY, 0, -1)
    result = []
    for pid in ids:
        raw = _redis.hget(PATIENTS_KEY, pid)
        if raw:
            d = json.loads(raw)
            d['id'] = int(pid)
            result.append(d)
    return result


def remove_patient(patient_id: int):
    _redis.zrem(QUEUE_KEY, str(patient_id))
    _redis.hdel(PATIENTS_KEY, str(patient_id))


def update_status(patient_id: int, status: str):
    raw = _redis.hget(PATIENTS_KEY, str(patient_id))
    if raw:
        d = json.loads(raw)
        d['status'] = status
        _redis.hset(PATIENTS_KEY, str(patient_id), json.dumps(d, ensure_ascii=False))


def get_queue_size() -> int:
    return _redis.zcard(QUEUE_KEY)


def clear_queue():
    _redis.delete(QUEUE_KEY, PATIENTS_KEY)
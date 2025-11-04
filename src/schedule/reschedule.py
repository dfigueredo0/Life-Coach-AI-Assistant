from datetime import datetime, timedelta

def detect_conflicts(events):
    events.sort(key=lambda e: e['start'])
    conflicts = []
    for i in range(len(events) - 1):
        if events[i]['end'] > events[i + 1]['start']:
            conflicts.append((events[i], events[i + 1]))
    return conflicts

def validate_confirm(events, new_event):
    for e in events:
        if e['start'] < new_event['end'] and new_event['start'] < e['end']:
            raise ValueError("CalendarConflictError: Overlapping events detected.")
    return True
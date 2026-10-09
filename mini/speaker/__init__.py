"""Speaker marking: who said each part of the transcript (doctor / patient or family / unknown). See SPEC 4.3."""
DOCTOR, OTHER, UNKNOWN = 'doctor', 'other', 'unknown'
LABEL_NAMES = {DOCTOR: '醫師', OTHER: '患者或家屬', UNKNOWN: '不明'}
TEXT_SOURCES = ('text', 'text2')      # labels read from the text (first / second text fill), not heard: shown with a star

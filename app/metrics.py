from prometheus_client import Histogram, Counter

infer_total = Counter(
    "faces_infer_total",
    "Infer requests processed",
    ["status"],
)
infer_faces_detected = Counter(
    "faces_detected_total",
    "Faces detected across infer requests",
)
infer_clusters_created = Counter(
    "faces_clusters_created_total",
    "Clusters created across infer requests",
)
infer_occurrences_written = Counter(
    "faces_occurrences_written_total",
    "Face occurrences persisted across infer requests",
)
infer_duration = Histogram(
    "faces_infer_duration_seconds",
    "Duration of an infer request",
)
reader_requests = Counter(
    "faces_reader_requests_total",
    "Reader REST requests",
    ["endpoint", "status"],
)
cascade_total = Counter(
    "faces_cascade_total",
    "Cascade delete requests",
    ["status"],
)
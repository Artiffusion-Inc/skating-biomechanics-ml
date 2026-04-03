"""DeepSORT трекер (заглушка для будущей интеграции)."""

# TODO: Интегрировать deep-sort-realtime для appearance-based ReID
# Это требует дополнительную зависимость: pip install deep-sort-realtime


class DeepSORTTracker:
    """Заглушка для будущего DeepSORT трекера."""

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "DeepSORTTracker еще не реализован. Используйте Sports2DTracker."
        )

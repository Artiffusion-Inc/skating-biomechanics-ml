"""Персистентность настроек UI.

Settings persistence for UI configuration.
"""

import json
from pathlib import Path


class UIConfig:
    """Загрузка и сохранение настроек UI.

    Load/save UI settings to ~/.config/skating-ui/settings.json
    """

    DEFAULT_CONFIG_PATH = Path.home() / ".config" / "skating-ui" / "settings.json"

    # Default settings
    DEFAULT_SETTINGS = {
        "layers": {
            "skeleton": True,
            "velocity": True,
            "trails": True,
            "edge_indicators": True,
            "subtitles": True,
        },
        "advanced": {
            "enable_3d": False,
            "blade_3d": False,
            "com_trajectory": False,
            "floor_mode": False,
            "no_3d_autoscale": False,
        },
        "parameters": {
            "trail_length": 20,
            "d_3d_scale": 0.6,
            "font_size": 30,
        },
    }

    def __init__(self, config_path: Path | None = None) -> None:
        """Инициализация с путём к конфигу.

        Args:
            config_path: Путь к файлу настроек. Если None, используется дефолтный.
        """
        self._config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._ensure_config_dir()
        self._settings = self._load()

    def _ensure_config_dir(self) -> None:
        """Создать директорию для конфига если не существует."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        """Загрузить настройки из файла."""
        if self._config_path.exists():
            try:
                with open(self._config_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                # Merge with defaults to handle new keys
                return self._merge_defaults(loaded)
            except (OSError, json.JSONDecodeError):
                return self.DEFAULT_SETTINGS.copy()
        return self.DEFAULT_SETTINGS.copy()

    def _merge_defaults(self, loaded: dict) -> dict:
        """Объединить загруженные настройки с дефолтными."""
        result = self.DEFAULT_SETTINGS.copy()
        for section in result:
            if section in loaded:
                result[section].update(loaded[section])
        return result

    def save(self, settings: dict) -> None:
        """Сохранить настройки в файл.

        Args:
            settings: Словарь с настройками для сохранения.
        """
        self._settings = self._merge_defaults(settings)
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._settings, f, indent=2, ensure_ascii=False)

    def load(self) -> dict:
        """Получить текущие настройки.

        Returns:
            Словарь с текущими настройками.
        """
        return self._settings.copy()

    def get_layer_settings(self) -> dict:
        """Получить настройки слоёв."""
        return self._settings.get("layers", {}).copy()

    def get_advanced_settings(self) -> dict:
        """Получить расширенные настройки."""
        return self._settings.get("advanced", {}).copy()

    def get_parameters(self) -> dict:
        """Получить параметры."""
        return self._settings.get("parameters", {}).copy()

    def update_layer(self, layer_name: str, value: bool) -> None:
        """Обновить настройки слоя.

        Args:
            layer_name: Имя слоя (skeleton, velocity, etc.)
            value: Новое значение (True/False)
        """
        if "layers" not in self._settings:
            self._settings["layers"] = {}
        self._settings["layers"][layer_name] = value
        self.save(self._settings)

    def update_parameter(self, param_name: str, value: int | float) -> None:
        """Обновить параметр.

        Args:
            param_name: Имя параметра (trail_length, d_3d_scale, etc.)
            value: Новое значение
        """
        if "parameters" not in self._settings:
            self._settings["parameters"] = {}
        self._settings["parameters"][param_name] = value
        self.save(self._settings)

    def update_advanced(self, key: str, value: bool) -> None:
        """Обновить расширенную настройку.

        Args:
            key: Ключ настройки (enable_3d, blade_3d, etc.)
            value: Новое значение
        """
        if "advanced" not in self._settings:
            self._settings["advanced"] = {}
        self._settings["advanced"][key] = value
        self.save(self._settings)

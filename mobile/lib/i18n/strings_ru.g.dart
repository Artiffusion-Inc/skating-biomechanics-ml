///
/// Generated file. Do not edit.
///
// coverage:ignore-file
// ignore_for_file: type=lint, unused_import
// dart format off

part of 'strings.g.dart';

// Path: <root>
typedef TranslationsRu = Translations; // ignore: unused_element
class Translations with BaseTranslations<AppLocale, Translations> {
	/// Returns the current translations of the given [context].
	///
	/// Usage:
	/// final t = Translations.of(context);
	static Translations of(BuildContext context) => InheritedLocaleData.of<AppLocale, Translations>(context).translations;

	/// You can call this constructor and build your own translation instance of this locale.
	/// Constructing via the enum [AppLocale.build] is preferred.
	Translations({Map<String, Node>? overrides, PluralResolver? cardinalResolver, PluralResolver? ordinalResolver, TranslationMetadata<AppLocale, Translations>? meta})
		: assert(overrides == null, 'Set "translation_overrides: true" in order to enable this feature.'),
		  $meta = meta ?? TranslationMetadata(
		    locale: AppLocale.ru,
		    overrides: overrides ?? {},
		    cardinalResolver: cardinalResolver,
		    ordinalResolver: ordinalResolver,
		  ) {
		$meta.setFlatMapFunction(_flatMapFunction);
	}

	/// Metadata for the translations of <ru>.
	@override final TranslationMetadata<AppLocale, Translations> $meta;

	/// Access flat map
	dynamic operator[](String key) => $meta.getTranslation(key);

	late final Translations _root = this; // ignore: unused_field

	Translations $copyWith({TranslationMetadata<AppLocale, Translations>? meta}) => Translations(meta: meta ?? this.$meta);

	// Translations
	late final TranslationsAppRu app = TranslationsAppRu.internal(_root);
	late final TranslationsPermissionsRu permissions = TranslationsPermissionsRu.internal(_root);
	late final TranslationsBleRu ble = TranslationsBleRu.internal(_root);
	late final TranslationsCalibrationRu calibration = TranslationsCalibrationRu.internal(_root);
	late final TranslationsCameraRu camera = TranslationsCameraRu.internal(_root);
	late final TranslationsCaptureRu capture = TranslationsCaptureRu.internal(_root);
	late final TranslationsMetricsRu metrics = TranslationsMetricsRu.internal(_root);
	late final TranslationsExportRu export = TranslationsExportRu.internal(_root);
}

// Path: app
class TranslationsAppRu {
	TranslationsAppRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'EdgeSense Capture'
	String get title => 'EdgeSense Capture';
}

// Path: permissions
class TranslationsPermissionsRu {
	TranslationsPermissionsRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Предоставить разрешения'
	String get grant => 'Предоставить разрешения';

	/// ru: 'Для BLE сканирования нужно разрешение на местоположение'
	String get bleRequired => 'Для BLE сканирования нужно разрешение на местоположение';

	/// ru: 'Для работы нужны разрешения'
	String get required => 'Для работы нужны разрешения';

	/// ru: 'Bluetooth, Камера, Микрофон, Местоположение'
	String get list => 'Bluetooth, Камера, Микрофон, Местоположение';
}

// Path: ble
class TranslationsBleRu {
	TranslationsBleRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Подключение IMU'
	String get scanTitle => 'Подключение IMU';

	/// ru: 'Сканировать'
	String get scan => 'Сканировать';

	/// ru: 'Сканировать снова'
	String get rescan => 'Сканировать снова';

	/// ru: 'Далее'
	String get next => 'Далее';

	/// ru: 'Bluetooth выключен'
	String get bluetoothOff => 'Bluetooth выключен';

	/// ru: 'Нажмите на устройство для назначения'
	String get assignHint => 'Нажмите на устройство для назначения';

	/// ru: 'Левый'
	String get left => 'Левый';

	/// ru: 'Правый'
	String get right => 'Правый';

	/// ru: 'оба подключены'
	String get bothConnected => 'оба подключены';

	/// ru: 'можно продолжить'
	String get oneConnected => 'можно продолжить';

	/// ru: 'подключение…'
	String get connecting => 'подключение…';

	/// ru: 'Настройки датчика'
	String get sensorSettings => 'Настройки датчика';

	late final TranslationsBleBatteryRu battery = TranslationsBleBatteryRu.internal(_root);
	late final TranslationsBleReturnRateRu returnRate = TranslationsBleReturnRateRu.internal(_root);
	late final TranslationsBleRenameRu rename = TranslationsBleRenameRu.internal(_root);
}

// Path: calibration
class TranslationsCalibrationRu {
	TranslationsCalibrationRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Калибровка'
	String get title => 'Калибровка';

	/// ru: 'Удерживайте датчики неподвижно для калибровки нулевого угла'
	String get instruction => 'Удерживайте датчики неподвижно\nдля калибровки нулевого угла';

	/// ru: 'Левый'
	String get left => 'Левый';

	/// ru: 'Правый'
	String get right => 'Правый';

	/// ru: 'Начать запись'
	String get start => 'Начать запись';

	/// ru: 'Калибровать'
	String get calibrate => 'Калибровать';

	/// ru: 'Пропустить'
	String get skip => 'Пропустить';

	/// ru: 'Калибровка завершена'
	String get done => 'Калибровка завершена';

	/// ru: 'Стойте неподвижно...'
	String get running => 'Стойте неподвижно...';

	/// ru: 'Ошибка подключения'
	String get errorPrefix => 'Ошибка подключения';

	/// ru: 'нет данных'
	String get noData => 'нет данных';

	/// ru: 'Начать запись'
	String get startCapture => 'Начать запись';
}

// Path: camera
class TranslationsCameraRu {
	TranslationsCameraRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Камера'
	String get title => 'Камера';

	/// ru: 'Начать запись'
	String get startCapture => 'Начать запись';

	/// ru: 'Камера недоступна'
	String get unavailable => 'Камера недоступна';

	/// ru: 'Повторить'
	String get retry => 'Повторить';

	/// ru: 'Разрешение'
	String get resolution => 'Разрешение';

	/// ru: 'Блокировка ориентации'
	String get orientation => 'Блокировка ориентации';

	/// ru: 'Портретный режим'
	String get portrait => 'Портретный режим';

	/// ru: 'Сетка'
	String get grid => 'Сетка';

	/// ru: 'Настройки'
	String get settings => 'Настройки';

	/// ru: 'Датчики'
	String get sensors => 'Датчики';

	/// ru: 'Настройки камеры'
	String get settingsTitle => 'Настройки камеры';

	late final TranslationsCameraResolutionsRu resolutions = TranslationsCameraResolutionsRu.internal(_root);
}

// Path: capture
class TranslationsCaptureRu {
	TranslationsCaptureRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Запись...'
	String get recording => 'Запись...';

	/// ru: 'Стоп'
	String get stop => 'Стоп';

	/// ru: 'Сохранение...'
	String get saving => 'Сохранение...';

	/// ru: 'EdgeSense Capture'
	String get shareText => 'EdgeSense Capture';
}

// Path: metrics
class TranslationsMetricsRu {
	TranslationsMetricsRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Датчики'
	String get title => 'Датчики';

	/// ru: 'Гироскоп'
	String get gyro => 'Гироскоп';

	/// ru: 'Акселерометр'
	String get accel => 'Акселерометр';

	/// ru: 'Edge Angle'
	String get edgeAngle => 'Edge Angle';
}

// Path: export
class TranslationsExportRu {
	TranslationsExportRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Результат'
	String get title => 'Результат';

	/// ru: 'Запись сохранена'
	String get success => 'Запись сохранена';

	/// ru: 'Ошибка при сохранении'
	String get error => 'Ошибка при сохранении';

	/// ru: 'Новая запись'
	String get newCapture => 'Новая запись';
}

// Path: ble.battery
class TranslationsBleBatteryRu {
	TranslationsBleBatteryRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Заряд батареи'
	String get title => 'Заряд батареи';

	/// ru: 'Неизвестно — запросите'
	String get unknown => 'Неизвестно — запросите';

	/// ru: 'Запросить'
	String get request => 'Запросить';
}

// Path: ble.returnRate
class TranslationsBleReturnRateRu {
	TranslationsBleReturnRateRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Частота передачи'
	String get title => 'Частота передачи';

	/// ru: 'Гц (после записи переподключить)'
	String get hint => 'Гц (после записи переподключить)';

	/// ru: 'Выбрать'
	String get select => 'Выбрать';
}

// Path: ble.rename
class TranslationsBleRenameRu {
	TranslationsBleRenameRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: 'Переименовать (ID 0-255)'
	String get title => 'Переименовать (ID 0-255)';

	/// ru: 'Изменить'
	String get action => 'Изменить';

	/// ru: 'Переименовать датчик'
	String get dialogTitle => 'Переименовать датчик';

	/// ru: 'Новый ID (0-255)'
	String get label => 'Новый ID (0-255)';

	/// ru: 'Например: 1'
	String get placeholder => 'Например: 1';

	/// ru: 'Отмена'
	String get cancel => 'Отмена';

	/// ru: 'Сохранить'
	String get save => 'Сохранить';
}

// Path: camera.resolutions
class TranslationsCameraResolutionsRu {
	TranslationsCameraResolutionsRu.internal(this._root);

	final Translations _root; // ignore: unused_field

	// Translations

	/// ru: '320×240 (низкое)'
	String get low => '320×240 (низкое)';

	/// ru: '720×480 (среднее)'
	String get medium => '720×480 (среднее)';

	/// ru: '1280×720 (HD)'
	String get high => '1280×720 (HD)';

	/// ru: '1920×1080 (Full HD)'
	String get veryHigh => '1920×1080 (Full HD)';

	/// ru: '3840×2160 (4K)'
	String get ultraHigh => '3840×2160 (4K)';

	/// ru: 'Максимальное'
	String get max => 'Максимальное';
}

/// The flat map containing all translations for locale <ru>.
/// Only for edge cases! For simple maps, use the map function of this library.
///
/// The Dart AOT compiler has issues with very large switch statements,
/// so the map is split into smaller functions (512 entries each).
extension on Translations {
	dynamic _flatMapFunction(String path) {
		return switch (path) {
			'app.title' => 'EdgeSense Capture',
			'permissions.grant' => 'Предоставить разрешения',
			'permissions.bleRequired' => 'Для BLE сканирования нужно разрешение на местоположение',
			'permissions.required' => 'Для работы нужны разрешения',
			'permissions.list' => 'Bluetooth, Камера, Микрофон, Местоположение',
			'ble.scanTitle' => 'Подключение IMU',
			'ble.scan' => 'Сканировать',
			'ble.rescan' => 'Сканировать снова',
			'ble.next' => 'Далее',
			'ble.bluetoothOff' => 'Bluetooth выключен',
			'ble.assignHint' => 'Нажмите на устройство для назначения',
			'ble.left' => 'Левый',
			'ble.right' => 'Правый',
			'ble.bothConnected' => 'оба подключены',
			'ble.oneConnected' => 'можно продолжить',
			'ble.connecting' => 'подключение…',
			'ble.sensorSettings' => 'Настройки датчика',
			'ble.battery.title' => 'Заряд батареи',
			'ble.battery.unknown' => 'Неизвестно — запросите',
			'ble.battery.request' => 'Запросить',
			'ble.returnRate.title' => 'Частота передачи',
			'ble.returnRate.hint' => 'Гц (после записи переподключить)',
			'ble.returnRate.select' => 'Выбрать',
			'ble.rename.title' => 'Переименовать (ID 0-255)',
			'ble.rename.action' => 'Изменить',
			'ble.rename.dialogTitle' => 'Переименовать датчик',
			'ble.rename.label' => 'Новый ID (0-255)',
			'ble.rename.placeholder' => 'Например: 1',
			'ble.rename.cancel' => 'Отмена',
			'ble.rename.save' => 'Сохранить',
			'calibration.title' => 'Калибровка',
			'calibration.instruction' => 'Удерживайте датчики неподвижно\nдля калибровки нулевого угла',
			'calibration.left' => 'Левый',
			'calibration.right' => 'Правый',
			'calibration.start' => 'Начать запись',
			'calibration.calibrate' => 'Калибровать',
			'calibration.skip' => 'Пропустить',
			'calibration.done' => 'Калибровка завершена',
			'calibration.running' => 'Стойте неподвижно...',
			'calibration.errorPrefix' => 'Ошибка подключения',
			'calibration.noData' => 'нет данных',
			'calibration.startCapture' => 'Начать запись',
			'camera.title' => 'Камера',
			'camera.startCapture' => 'Начать запись',
			'camera.unavailable' => 'Камера недоступна',
			'camera.retry' => 'Повторить',
			'camera.resolution' => 'Разрешение',
			'camera.orientation' => 'Блокировка ориентации',
			'camera.portrait' => 'Портретный режим',
			'camera.grid' => 'Сетка',
			'camera.settings' => 'Настройки',
			'camera.sensors' => 'Датчики',
			'camera.settingsTitle' => 'Настройки камеры',
			'camera.resolutions.low' => '320×240 (низкое)',
			'camera.resolutions.medium' => '720×480 (среднее)',
			'camera.resolutions.high' => '1280×720 (HD)',
			'camera.resolutions.veryHigh' => '1920×1080 (Full HD)',
			'camera.resolutions.ultraHigh' => '3840×2160 (4K)',
			'camera.resolutions.max' => 'Максимальное',
			'capture.recording' => 'Запись...',
			'capture.stop' => 'Стоп',
			'capture.saving' => 'Сохранение...',
			'capture.shareText' => 'EdgeSense Capture',
			'metrics.title' => 'Датчики',
			'metrics.gyro' => 'Гироскоп',
			'metrics.accel' => 'Акселерометр',
			'metrics.edgeAngle' => 'Edge Angle',
			'export.title' => 'Результат',
			'export.success' => 'Запись сохранена',
			'export.error' => 'Ошибка при сохранении',
			'export.newCapture' => 'Новая запись',
			_ => null,
		};
	}
}

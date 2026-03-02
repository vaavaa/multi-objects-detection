# Локали для language negotiation (BCP 47).
# Если переданный lang не в списке — вернётся 400 или fallback на DEFAULT_LOCALE.

DEFAULT_LOCALE = "ru-RU"

ALLOWED_LOCALES = [
    "ru-RU",
    "en-US",
    "en-GB",
    "de-DE",
    "fr-FR",
    "es-ES",
    "uk-UA",
    "kk-KZ",
]

# Ключи текстов для HTML-ответов (заголовки страниц, подписи единиц).
# Использование: t(locale, "page_title_current_local_time") и т.д.
TEXTS = {
    # Заголовки страниц (title + h1)
    "page_title_current_local_time": {
        "ru-RU": "Текущее локальное время",
        "en-US": "Current Local Time",
        "en-GB": "Current Local Time",
        "de-DE": "Aktuelle Ortszeit",
        "fr-FR": "Heure locale actuelle",
        "es-ES": "Hora local actual",
        "uk-UA": "Поточний місцевий час",
        "kk-KZ": "Жергілікті уақыт",
    },
    "page_title_formatted_time": {
        "ru-RU": "Форматированное время",
        "en-US": "Formatted Time",
        "en-GB": "Formatted Time",
        "de-DE": "Formatierte Zeit",
        "fr-FR": "Heure formatée",
        "es-ES": "Hora formateada",
        "uk-UA": "Відформатований час",
        "kk-KZ": "Пішімделген уақыт",
    },
    "page_title_converted_time": {
        "ru-RU": "Преобразованное время",
        "en-US": "Converted Time",
        "en-GB": "Converted Time",
        "de-DE": "Umgewandelte Zeit",
        "fr-FR": "Heure convertie",
        "es-ES": "Hora convertida",
        "uk-UA": "Перетворений час",
        "kk-KZ": "Түрлендірілген уақыт",
    },
    "page_title_parsed_timestamp_utc": {
        "ru-RU": "Распознанная метка времени (UTC)",
        "en-US": "Parsed Timestamp (UTC)",
        "en-GB": "Parsed Timestamp (UTC)",
        "de-DE": "Geparste Zeitangabe (UTC)",
        "fr-FR": "Horodatage analysé (UTC)",
        "es-ES": "Marca de tiempo parseada (UTC)",
        "uk-UA": "Розпізнана мітка часу (UTC)",
        "kk-KZ": "Танылған уақыт белгісі (UTC)",
    },
    "page_title_elapsed_time": {
        "ru-RU": "Прошедшее время",
        "en-US": "Elapsed Time",
        "en-GB": "Elapsed Time",
        "de-DE": "Vergangene Zeit",
        "fr-FR": "Temps écoulé",
        "es-ES": "Tiempo transcurrido",
        "uk-UA": "Минулий час",
        "kk-KZ": "Өткен уақыт",
    },
    "page_title_valid_time_zones": {
        "ru-RU": "Допустимые часовые пояса",
        "en-US": "Valid Time Zones",
        "en-GB": "Valid Time Zones",
        "de-DE": "Gültige Zeitzonen",
        "fr-FR": "Fuseaux horaires valides",
        "es-ES": "Zonas horarias válidas",
        "uk-UA": "Допустимі часові пояси",
        "kk-KZ": "Жарамды уақыт белдеулері",
    },
    # Единицы для elapsed_time (подпись в строке результата: "X seconds" / "X секунд")
    "unit_seconds": {
        "ru-RU": "секунд",
        "en-US": "seconds",
        "en-GB": "seconds",
        "de-DE": "Sekunden",
        "fr-FR": "secondes",
        "es-ES": "segundos",
        "uk-UA": "секунд",
        "kk-KZ": "секунд",
    },
    "unit_minutes": {
        "ru-RU": "минут",
        "en-US": "minutes",
        "en-GB": "minutes",
        "de-DE": "Minuten",
        "fr-FR": "minutes",
        "es-ES": "minutos",
        "uk-UA": "хвилин",
        "kk-KZ": "минут",
    },
    "unit_hours": {
        "ru-RU": "часов",
        "en-US": "hours",
        "en-GB": "hours",
        "de-DE": "Stunden",
        "fr-FR": "heures",
        "es-ES": "horas",
        "uk-UA": "годин",
        "kk-KZ": "сағат",
    },
    "unit_days": {
        "ru-RU": "дней",
        "en-US": "days",
        "en-GB": "days",
        "de-DE": "Tage",
        "fr-FR": "jours",
        "es-ES": "días",
        "uk-UA": "днів",
        "kk-KZ": "күн",
    },
}


def t(locale: str, key: str) -> str:
    """Возвращает перевод по ключу для локали; при отсутствии — fallback на DEFAULT_LOCALE, затем ключ."""
    if locale not in ALLOWED_LOCALES:
        locale = DEFAULT_LOCALE
    return TEXTS.get(key, {}).get(locale) or TEXTS.get(key, {}).get(DEFAULT_LOCALE) or key

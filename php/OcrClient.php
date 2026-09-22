<?php

declare(strict_types=1);

namespace App\Libraries;

use RuntimeException;

/**
 * Клиент локального OCR-сервиса (kzru-ocr).
 *
 * Вся логика распознавания живёт в сервисе; здесь только транспорт, ретраи и
 * разбор контракта. Сервис слушает на loopback, поэтому ни аутентификации, ни
 * TLS не предполагается — вынос порта наружу запрещён конфигурацией сети.
 *
 * Использование в контроллере CodeIgniter 4:
 *
 *     $client = new OcrClient();
 *     $result = $client->recognize($pathToPdf);
 *     $text   = $result['text'];
 *     foreach ($result['warnings'] as $w) {
 *         // $w['type'] === 'bin_checksum_failed' → показать «проверьте вручную»
 *     }
 */
final class OcrClient
{
    private string $baseUrl;
    private int $timeout;
    private int $connectTimeout;
    private int $retries;

    public function __construct(
        ?string $baseUrl = null,
        int $timeout = 180,
        int $connectTimeout = 3,
        int $retries = 2
    ) {
        $this->baseUrl = rtrim($baseUrl ?? (getenv('OCR_SERVICE_URL') ?: 'http://127.0.0.1:8099'), '/');
        $this->timeout = $timeout;
        $this->connectTimeout = $connectTimeout;
        $this->retries = $retries;
    }

    /**
     * Синхронное распознавание.
     *
     * Профили запрашиваются у сервиса (`readiness()['profiles']`), а не
     * зашиты здесь: набор профилей меняется вместе с сервисом.
     *
     * @return array{text:string,pages:array,fields:array,warnings:array,elapsed_s:float,mean_conf:float}
     */
    public function recognize(string $pdfPath, string $profile = 'balanced', string $render = 'default'): array
    {
        if (!is_readable($pdfPath)) {
            throw new RuntimeException("PDF недоступен для чтения: {$pdfPath}");
        }

        $url = $this->baseUrl . '/ocr?profile=' . urlencode($profile) . '&render=' . urlencode($render);

        $lastError = '';
        for ($attempt = 0; $attempt <= $this->retries; $attempt++) {
            try {
                return $this->post($url, $pdfPath);
            } catch (RuntimeException $e) {
                $lastError = $e->getMessage();
                // 4xx — ошибка запроса, повтор бессмыслен.
                if (str_starts_with($lastError, 'HTTP 4')) {
                    throw $e;
                }
                if ($attempt < $this->retries) {
                    usleep(250_000 * (int) pow(2, $attempt));
                }
            }
        }

        throw new RuntimeException("OCR-сервис недоступен после повторов: {$lastError}");
    }

    /**
     * Готовность сервиса. Бросает исключение, если сервис не готов (503),
     * поэтому подходит для проверки перед пакетной обработкой.
     */
    public function readiness(): array
    {
        return $this->get('/readyz');
    }

    /** Живость процесса. Не проверяет зависимости — только что сервис отвечает. */
    public function health(): array
    {
        return $this->get('/healthz');
    }

    /**
     * Значения поля, пригодные для автоматического разбора.
     *
     * Для `bin` это статусы `valid` и `repaired`, для `dates` — только
     * календарно валидные. Всё остальное требует ручной проверки и намеренно
     * не возвращается: исправленный или непроверенный номер не должен
     * выглядеть подтверждённым.
     *
     * @param array $fields блок `fields` из ответа сервиса
     * @return string[]
     */
    public static function confirmedValues(array $fields, string $key): array
    {
        $items = $fields[$key] ?? [];
        $out = [];
        foreach ($items as $item) {
            $status = $item['status'] ?? null;
            if ($key === 'bin' && !in_array($status, ['valid', 'repaired'], true)) {
                continue;
            }
            if ($key === 'dates' && $status !== 'valid') {
                continue;
            }
            if (($item['value'] ?? null) !== null) {
                $out[] = (string) $item['value'];
            }
        }

        return $out;
    }

    /**
     * Поля, которые обязан посмотреть оператор: исправленные и непроверенные
     * номера, несуществующие даты, суммы с расхождением прописи.
     *
     * @return array<int, array{field:string, item:array}>
     */
    public static function needsReview(array $fields): array
    {
        $out = [];
        foreach ($fields['bin'] ?? [] as $item) {
            if (($item['requires_review'] ?? false) === true) {
                $out[] = ['field' => 'bin', 'item' => $item];
            }
        }
        foreach ($fields['dates'] ?? [] as $item) {
            if (($item['status'] ?? null) === 'invalid') {
                $out[] = ['field' => 'dates', 'item' => $item];
            }
        }
        foreach ($fields['amounts'] ?? [] as $item) {
            if (($item['words_match'] ?? null) === false) {
                $out[] = ['field' => 'amounts', 'item' => $item];
            }
        }

        return $out;
    }

    private function get(string $path): array
    {
        $ch = curl_init($this->baseUrl . $path);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 5,
            CURLOPT_CONNECTTIMEOUT => $this->connectTimeout,
        ]);
        $body = curl_exec($ch);
        $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $err  = curl_error($ch);
        curl_close($ch);

        if ($body === false) {
            throw new RuntimeException("{$path} недоступен: {$err}");
        }
        if ($code !== 200) {
            // 503 на /readyz — это штатный ответ «не готов», а не сбой связи.
            throw new RuntimeException("HTTP {$code} на {$path}: " . substr((string) $body, 0, 200));
        }

        return $this->decode((string) $body);
    }

    private function post(string $url, string $pdfPath): array
    {
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST           => true,
            CURLOPT_TIMEOUT        => $this->timeout,
            CURLOPT_CONNECTTIMEOUT => $this->connectTimeout,
            CURLOPT_POSTFIELDS     => [
                'file' => new \CURLFile($pdfPath, 'application/pdf', basename($pdfPath)),
            ],
        ]);

        $body = curl_exec($ch);
        $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $err  = curl_error($ch);
        curl_close($ch);

        if ($body === false) {
            throw new RuntimeException("curl: {$err}");
        }
        if ($code !== 200) {
            $detail = '';
            $decoded = json_decode((string) $body, true);
            if (is_array($decoded) && isset($decoded['detail'])) {
                $detail = is_string($decoded['detail']) ? $decoded['detail'] : json_encode($decoded['detail']);
            }
            throw new RuntimeException("HTTP {$code}: {$detail}");
        }

        return $this->decode((string) $body);
    }

    private function decode(string $body): array
    {
        $data = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
        if (!is_array($data)) {
            throw new RuntimeException('сервис вернул не JSON-объект');
        }

        return $data;
    }
}
